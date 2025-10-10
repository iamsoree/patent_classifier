# util/lm_studio.py

import streamlit as st
from component.model_list import show_model_list
import requests
import time
from util.data_processor import DataProcessor
import pandas as pd

def settings():

    with st.expander("**LM STUDIO SETTING**", expanded = True):

        col_api1, col_api2 = st.columns(2)

        with col_api1:
            api_url_input = st.text_input(
                "**BASE URL**", 
                value = st.session_state.get("api_url", "http://localhost:1234/v1/chat/completions")
            )
            st.session_state.api_url = api_url_input
            
        with col_api2:
            api_model_input = st.text_input(
                "**MODEL**", 
                value = st.session_state.get("api_model", "qwen/qwen3-14b")
            )
            st.session_state.api_model = api_model_input

        col_left, col_right = st.columns([1, 7])

        with col_left:
            show_model_list(key_prefix = "lm_studio")

        with col_right:
            connection_click = st.button("**API CONNECTION**")

        if connection_click:

            try:

                test_response = requests.post(
                    st.session_state.api_url,
                    json = {
                        "model": st.session_state.api_model,
                        "messages": [{"role": "user", "content": "Hello!"}],
                        "max_tokens": 10
                    },
                    timeout = 30
                )

                if test_response.status_code != 200:
                    st.info(f"🔴 API CONNECTION FAILED ({test_response.status_code})")
                    st.session_state.api_connection_success = False
                    return
                
                response_data = test_response.json()

                actual_model = response_data.get("model", "")

                if actual_model != st.session_state.api_model:
                    st.info(
                        f"""🔴 API CONNECTION FAILED  
                        - REQUEST MODEL : {st.session_state.api_model}  
                        - LOAD MODEL : {actual_model}"""
                    )
                    st.session_state.api_connection_success = False
                    return

                st.info("🔵 API CONNECTION SUCCESSFUL")

                st.session_state.api_connection_success = True

            except Exception as e:
                st.info(f"🔴 API CONNECTION FAILED : {e}")
                st.session_state.api_connection_success = False

def get_score_for_candidate(text, code, desc, step1_prompt, api_url, api_model):

    prompt = step1_prompt.format(text = text, code = code, desc = desc)
    
    try:

        response = requests.post(
            api_url,
            json = {
                "model": api_model,
                "messages": [
                    {"role": "system", "content": "당신은 특허 문서를 정해진 코드로 분류하는 전문가입니다."},
                    {"role": "user", "content": prompt}
                ],
                "temperature": 0,
                "max_tokens": 8
            },
            timeout = 30
        )

        if response.status_code != 200:
            return None
        
        result = response.json()
        score_text = result['choices'][0]['message']['content'].strip()
        
        try:
            score = float(score_text)
            return max(0.0, min(score, 10.0))
        except ValueError:
            return None
            
    except Exception:
        return None

def reselect_best_code(text, candidate_codes, cpc_candidates, step2_prompt, api_url, api_model):

    candidate_text = "\n".join([f"{code} : {cpc_candidates[code]}" for code in candidate_codes])

    prompt = step2_prompt.format(text = text, candidate_text = candidate_text)
    
    try:

        response = requests.post(
            api_url,
            json = {
                "model": api_model,
                "messages": [
                    {"role": "system", "content": "당신은 특허 문서를 정해진 코드로 분류하는 전문가입니다."},
                    {"role": "user", "content": prompt}
                ],
                "temperature": 0,
                "max_tokens": 15
            },
            timeout = 30
        )
        
        if response.status_code == 200:
            result = response.json()
            final_choice = result['choices'][0]['message']['content'].strip()
            for code in candidate_codes:
                if code in final_choice:
                    return code
            return "ERROR"
        
        else:
            return "ERROR"
            
    except Exception:
        return "ERROR"

def inference(selected_columns, df, custom_separator, step1_prompt, step2_prompt, cpc_candidates):

    progress_bar = st.progress(0)
        
    with st.spinner("**CLASSIFICATION IN PROGRESS ...**"):

        if len(selected_columns) == 1:
            data_to_classify = df[selected_columns[0]].dropna().astype(str).tolist()
            
        else:
            clean_df = df[selected_columns].dropna()
            data_to_classify = clean_df.apply(
                lambda row: custom_separator.join([str(row[col]) for col in selected_columns]),
                axis = 1
            ).tolist()

        data_to_classify = [text for text in data_to_classify if text.strip()]
        
        if not data_to_classify:
            st.info("🔴 NO DATA AVAILABLE FOR CLASSIFICATION.")
        
        display_results = []
        full_results = []

        classifications = []
        
        for i, text in enumerate(data_to_classify):

            try:

                scores = {}

                for code, desc in cpc_candidates.items():
                    score = get_score_for_candidate(text, code, desc, step1_prompt, st.session_state.api_url, st.session_state.api_model)
                    if score is not None:
                        scores[code] = score
                    time.sleep(0.5)

                if scores:

                    max_score = max(scores.values())

                    candidates_with_max_score = [code for code, s in scores.items() if s == max_score]

                    if len(candidates_with_max_score) == 1:
                        classification = candidates_with_max_score[0]

                    else:
                        classification = reselect_best_code(text, candidates_with_max_score, cpc_candidates, step2_prompt, st.session_state.api_url, st.session_state.api_model)
                        time.sleep(0.5)

                else:
                    classification = "ERROR"

                classifications.append(classification)
                    
            except Exception:
                classifications.append("ERROR")
            
            progress_bar.progress((i + 1) / len(data_to_classify))

            time.sleep(1.0)

        available_cols = DataProcessor.get_available_columns(df)

        for idx, classification in enumerate(classifications):
            if idx < len(df):

                row = df.iloc[idx]

                display_results.append({
                    '출원 번호' : row.get('출원번호', ''),
                    '발명의 명칭' : row.get('발명의 명칭', ''),
                    '분류 코드' : classification
                })

                full_row = {col : row.get(col, '') for col in available_cols}

                full_row['분류 코드'] = classification

                full_results.append(full_row)

        results_df = pd.DataFrame(display_results)
        full_results_df = pd.DataFrame(full_results)
        
        st.session_state.classification_results = results_df
        st.session_state.classification_results_full = full_results_df
        
        st.toast("**CLASSIFICATION IS COMPLETE**")