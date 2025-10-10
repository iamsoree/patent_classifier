# component/prompt_engineering/prompt_setting.py

import streamlit as st
from util import lm_studio

def get_prompt_templates(language):
    
    if language == "한국어":

        step1_prompt = """다음은 특허 정보입니다.

{text}

아래의 분류 코드가 이 특허에 얼마나 적합한 지 평가해 주세요.  

0.0은 '전혀 관련 없음.'을 의미하고,  
5.0은 '부분적으로 관련 있음.'을 의미하며,  
10.0은 '완벽히 일치함.'을 의미합니다.  

분류 코드 : {code}
설명 : {desc}

0.0에서 10.0 사이의 소수점 1자리 숫자만 출력해 주세요.  
숫자 외에는 어떠한 단어도 출력하지 마세요."""

        step2_prompt = """다음은 특허 정보입니다.

{text}

아래의 후보 중에서 위의 특허에 가장 적합한 **단 1개의 분류 코드**를 골라 주세요.

[후보 및 설명]
{candidate_text}

규칙 :
1. 반드시 후보에 있는 분류 코드 중 하나만 출력하세요.
2. 후보에 없는 다른 분류 코드는 절대 출력하지 마세요.
3. 부가 설명이나 다른 단어 없이 오직 분류 코드만 출력하세요.
4. 출력 전에 신중히 검토하세요.
5. 출력 예시 : CPC_C01B"""

    else:

        step1_prompt = """The following is patent information.

{text}

Please evaluate how suitable the classification code below is for this patent.

0.0 means 'Not relevant at all.'
5.0 means 'Partially relevant.'
10.0 means 'Perfectly matches.'

Classification code : {code}
Description : {desc}

Output requirements :
- Output exactly one decimal number with one digit after the decimal point, between 0.0 and 10.0 inclusive.
- Do not output any words, explanations, punctuation, or extra characters — only the numeric value."""

        step2_prompt = """The following is patent information.

{text}

Please select **only one classification code** from the candidates below that best fits the patent.

[Candidates and Descriptions]
{candidate_text}

Rules :
1. Output exactly one classification code and nothing else.
2. The output must match one of the candidate codes exactly as provided (case-sensitive, no extra spaces or punctuation).
3. Do not output any additional words, explanations, or characters.
4. Review carefully before outputting.
5. Example output : CPC_C01B"""
    
    return step1_prompt, step2_prompt

def show(selected_columns, df, custom_separator):

    with st.expander("**PROMPT TEMPLATE**", expanded = False):

        col_lang, col_spacer = st.columns([1, 3])

        with col_lang:
            selected_language = st.selectbox(
                "**LANGUAGE**", 
                ["한국어", "ENGLISH"],
                key = "prompt_language"
            )
        
        default_step1, default_step2 = get_prompt_templates(selected_language)
        
        if 'current_language' not in st.session_state or st.session_state.current_language != selected_language:
            st.session_state.step1_prompt = default_step1
            st.session_state.step2_prompt = default_step2
            st.session_state.current_language = selected_language
        
        if 'step1_prompt' not in st.session_state:
            st.session_state.step1_prompt = default_step1
        if 'step2_prompt' not in st.session_state:
            st.session_state.step2_prompt = default_step2

        reset_suffix = f"_{st.session_state.get('reset_counter', 0)}"

        col1, col2 = st.columns(2)
        
        with col1:
            st.session_state.step1_prompt = st.text_area(
                "**[STEP 1] RELEVANCE SCORING**",
                value = st.session_state.step1_prompt,
                height = 350,
                help = "각 카테고리마다 적합도 점수를 계산합니다.",
                key = f"step1_textarea{reset_suffix}"
            )
        
        with col2:
            st.session_state.step2_prompt = st.text_area(
                "**[STEP 2] BEST MATCH SELECTION**", 
                value = st.session_state.step2_prompt,
                height = 350,
                help = "동점인 경우, 최적의 코드를 재선택합니다.",
                key = f"step2_textarea{reset_suffix}"
            )
        
        if st.button("**INITIALIZE PROMPT**"):
            init_step1, init_step2 = get_prompt_templates(selected_language)
            st.session_state.step1_prompt = init_step1
            st.session_state.step2_prompt = init_step2
            if 'reset_counter' not in st.session_state:
                st.session_state.reset_counter = 0
            st.session_state.reset_counter += 1
            st.rerun()

    step1_has_content = st.session_state.get('step1_prompt', '').strip() != ''
    step2_has_content = st.session_state.get('step2_prompt', '').strip() != ''

    return step1_has_content and step2_has_content