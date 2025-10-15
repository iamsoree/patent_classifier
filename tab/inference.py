# tab/inference.py

from dotenv import load_dotenv
import streamlit as st
import os
import glob
from util.database.ft_database import TrainingDatabase
import pickle
from util.model_detector import detect_model_type
from util.finetuning_inference import FineTuningInference
import time
from util.data_processor import DataProcessor
import pandas as pd
from util import excel_download

load_dotenv()

def show():

    with st.expander("**COLUMNS TO USE FOR INFERENCE**", expanded = True):

        df = st.session_state.uploaded_df

        selected_cols = st.multiselect(
            "**TEXT COLUMNS**",
            options = df.columns.tolist(),
            default = [col for col in ["발명의 명칭", "요약", "전체청구항"] if col in df.columns.tolist()],
            key = "inference_cols"
        )

    if not selected_cols:
        st.info("🔴 PLEASE SELECT AT LEAST ONE COLUMN.")

    else:

        with st.expander("**MODEL TO USE FOR INFERENCE**", expanded = True):

            model_selection_method = st.radio(
                "**MODEL SELECTION METHOD**",
                ["AUTOMATIC SEARCH", "MANUAL PATH ENTRY"],
                key = "model_selection_method"
            )

            if model_selection_method == "MANUAL PATH ENTRY":

                def on_manual_path_change():
                    new_path = st.session_state.get('manual_model_path', '')
                    if new_path and os.path.exists(new_path):
                        db = TrainingDatabase()
                        db_record = db.get_record_by_path(new_path)
                        if db_record:
                            st.session_state["inference_model_type_from_db"] = db_record.get('hyperparameters', {}).get('model_type', 'GENERATIVE')
                            st.session_state["inference_model_name_from_db"] = db_record.get('model_name', 'google/gemma-2-2b')
                            st.session_state["inference_model_type"] = db_record.get('hyperparameters', {}).get('model_type', 'GENERATIVE')
                            if "manual_model_type_override_inference" in st.session_state:
                                del st.session_state["manual_model_type_override_inference"]

                model_path = st.text_input(
                    "**ENTER THE PATH OF THE DESIRED MODEL**",
                    value = r"C:\company\wips\ft_llama_3.2_3b_1\merge_model",
                    help = "학습시킨 모델이 저장되어 있는 전체 경로를 입력하세요.",
                    key = "manual_model_path",
                    on_change = on_manual_path_change
                )

                if not model_path.strip():
                    st.info("🔴 MODEL PATH HAS NOT BEEN ENTERED YET.")
                    model_path = None

                else:
                    if os.path.exists(model_path):
                        db = TrainingDatabase()
                        db_record = db.get_record_by_path(model_path)
                        if db_record:
                            st.session_state["inference_model_type_from_db"] = db_record.get('hyperparameters', {}).get('model_type', 'GENERATIVE')
                            st.session_state["inference_model_name_from_db"] = db_record.get('model_name', 'google/gemma-2-2b')
                            if "manual_model_type_override_inference" not in st.session_state:
                                st.session_state["inference_model_type"] = db_record.get('hyperparameters', {}).get('model_type', 'GENERATIVE')

            else:

                base_dir = r"C:\company\wips"

                if os.path.exists(base_dir):

                    try:
                        # 생성 모델
                        merged_model_paths = glob.glob(os.path.join(base_dir, "*", "merge_model"))
                        # 표현 모델
                        direct_model_paths = []
                        for item in os.listdir(base_dir):
                            item_path = os.path.join(base_dir, item)
                            if os.path.isdir(item_path):
                                if not os.path.exists(os.path.join(item_path, "merge_model")):
                                    if os.path.exists(os.path.join(item_path, 'config.json')):
                                        direct_model_paths.append(item_path)

                        valid_models = []

                        for merged_path in merged_model_paths:  # 생성 모델
                            if (os.path.exists(os.path.join(merged_path, 'config.json')) or os.path.exists(os.path.join(merged_path, 'pytorch_model.bin')) or os.path.exists(os.path.join(merged_path, 'model.safetensors'))):
                                parent_dir = os.path.basename(os.path.dirname(merged_path))
                                valid_models.append((parent_dir, merged_path))

                        for direct_path in direct_model_paths:  # 표현 모델
                            if (os.path.exists(os.path.join(direct_path, 'pytorch_model.bin')) or os.path.exists(os.path.join(direct_path, 'model.safetensors'))):
                                folder_name = os.path.basename(direct_path)
                                valid_models.append((folder_name, direct_path))

                        if valid_models:

                            st.session_state['inference_valid_models'] = valid_models

                            def on_auto_model_select():
                                selected = st.session_state.get('auto_model_select', '-- SELECT --')
                                if selected != '-- SELECT --':
                                    valid_models_list = st.session_state.get('inference_valid_models', [])
                                    model_path = next((path for name, path in valid_models_list if name == selected), None)
                                    if model_path:
                                        db = TrainingDatabase()
                                        db_record = db.get_record_by_path(model_path)
                                        if db_record:
                                            st.session_state["inference_model_type_from_db"] = db_record.get('hyperparameters', {}).get('model_type', 'GENERATIVE')
                                            st.session_state["inference_model_name_from_db"] = db_record.get('model_name', 'google/gemma-2-2b')
                                            st.session_state["inference_model_type"] = db_record.get('hyperparameters', {}).get('model_type', 'GENERATIVE')
                                            if "manual_model_type_override_inference" in st.session_state:
                                                del st.session_state["manual_model_type_override_inference"]

                            model_options = ['-- SELECT --'] + [name for name, path in valid_models]

                            selected_model_name = st.selectbox(
                                "**SELECT ONE OF THE FOUND MODELS**",
                                options = model_options,
                                index = 0,
                                key = "auto_model_select",
                                on_change = on_auto_model_select
                            )

                            if selected_model_name == "-- SELECT --":
                                st.info("🔴 MODEL HAS NOT BEEN SELECTED YET.")
                                model_path = None
                            else:
                                model_path = next(path for name, path in valid_models if name == selected_model_name)
                                db = TrainingDatabase()
                                db_record = db.get_record_by_path(model_path)
                                if db_record:
                                    st.session_state["inference_model_type_from_db"] = db_record.get('hyperparameters', {}).get('model_type', 'GENERATIVE')
                                    st.session_state["inference_model_name_from_db"] = db_record.get('model_name', 'google/gemma-2-2b')
                                    if "manual_model_type_override_inference" not in st.session_state:
                                        st.session_state["inference_model_type"] = db_record.get('hyperparameters', {}).get('model_type', 'GENERATIVE')

                        else:
                            st.info("🔴 NO MODEL COULD BE FOUND USING AUTOMATIC SEARCH.")
                            model_path = None

                    except Exception as e:
                        st.info(f"🔴 {e}")
                        model_path = None

                else:
                    st.info(f"🔴 THE DEFAULT DIRECTORY DOES NOT EXIST. : {base_dir}")
                    model_path = None

            model_exists = False
            db_record = None
            db = TrainingDatabase()

            if model_path and os.path.exists(model_path):

                config_exists = os.path.exists(os.path.join(model_path, 'config.json'))
                model_file_exists = (os.path.exists(os.path.join(model_path, 'pytorch_model.bin')) or os.path.exists(os.path.join(model_path, 'model.safetensors')))

                if config_exists and model_file_exists:

                    model_exists = True

                    db_record = db.get_record_by_path(model_path)
                    # 표현 모델
                    label_file_path = os.path.join(model_path, 'label_mappings.pkl')
                    # 생성 모델
                    if not os.path.exists(label_file_path):
                        parent_dir = os.path.dirname(model_path)
                        label_file_path = os.path.join(parent_dir, 'label_mappings.pkl')

                    if os.path.exists(label_file_path):

                        try:
                            with open(label_file_path, 'rb') as f:
                                mappings = pickle.load(f)
                                model_labels = mappings['labels_list']
                                with st.expander("**LABELS FOR THE MODEL**", expanded = True):
                                    st.write(sorted(model_labels))

                        except Exception as e:
                            st.info(f"🔴 {e}")
                    else:
                        st.info("🔴 UNABLE TO FIND THE LABEL INFORMATION FILE.")
                else:
                    st.info("🔴 MODEL FILE DOES NOT EXIST.")
            elif model_path:
                st.info("🔴 NO MODEL FOUND AT THE SPECIFIED PATH.")

        if model_exists:
            
            with st.expander("**SETTING**", expanded = True):

                if db_record:
                    model_type_info = db_record.get('hyperparameters', {}).get('model_type', 'GENERATIVE')
                    st.info(f"🔵 {db_record.get('model_name')}&nbsp;&nbsp;|&nbsp;&nbsp;{model_type_info}&nbsp;&nbsp;|&nbsp;&nbsp;(MAX LENGTH) {db_record.get('max_length', 512)} / (STRIDE) {db_record.get('stride', 50)}")

                with st.expander("**TRANSFORMERS**", expanded = False):

                    col1, col2 = st.columns(2)

                    default_model_name = 'google/gemma-2-2b'

                    if db_record:
                        default_model_name = db_record.get('model_name', default_model_name)

                    if "inference_model_name_from_db" in st.session_state:
                        default_model_name = st.session_state["inference_model_name_from_db"]

                    with col1:
                        st.text_input(
                            "**HUGGING FACE TOKEN**", 
                            value = st.session_state.get('ft_hf_token') or os.getenv('HF_TOKEN'), 
                            type = "password", 
                            key = "ft_hf_token_inference"
                        )
                        
                    with col2:

                        def on_model_name_change():

                            new_model = st.session_state.get('ft_model_name_inference', '')
                            old_model = st.session_state.get('prev_model_name_for_detection_inference', '')
                            
                            if new_model != old_model:
                                detected = detect_model_type(new_model)
                                st.session_state["inference_model_type"] = detected
                                st.session_state["prev_model_name_for_detection_inference"] = new_model
                                # st.session_state["manual_model_type_override_inference"] = False
                                if "manual_model_type_override_inference" in st.session_state:
                                    del st.session_state["manual_model_type_override_inference"]

                        model_name = st.text_input(
                            "**MODEL NAME**", 
                            value = default_model_name, 
                            key = "ft_model_name_inference",
                            on_change = on_model_name_change
                        )

                    col3, col4 = st.columns(2)

                    with col4:

                        col_a, col_b = st.columns(2)

                        if model_name and model_name.strip():

                            if "inference_model_type_from_db" in st.session_state and "manual_model_type_override_inference" not in st.session_state:
                                st.session_state["inference_model_type"] = st.session_state["inference_model_type_from_db"]

                            elif "inference_model_type" not in st.session_state:
                                detected_type = detect_model_type(model_name)
                                st.session_state["inference_model_type"] = detected_type
                                st.session_state["prev_model_name_for_detection_inference"] = model_name

                            with col_a:
                                rep_clicked = st.button(
                                    "☑️ REPRESENTATION" if st.session_state.get("inference_model_type") == "REPRESENTATION" else "🟪 REPRESENTATION",
                                    width = 'stretch', key = "inference_btn_rep"
                                )
                                if rep_clicked:
                                    st.session_state["inference_model_type"] = "REPRESENTATION"
                                    st.session_state["manual_model_type_override_inference"] = True
                                    st.rerun()

                            with col_b:
                                gen_clicked = st.button(
                                    "☑️ GENERATIVE" if st.session_state.get("inference_model_type") == "GENERATIVE" else "🟪 GENERATIVE",
                                    width = 'stretch', key = "inference_btn_gen"
                                )
                                if gen_clicked:
                                    st.session_state["inference_model_type"] = "GENERATIVE"
                                    st.session_state["manual_model_type_override_inference"] = True
                                    st.rerun()

                with st.expander("**HYPERPARAMETER**", expanded = False):
                        
                    if db_record:
                        default_max_length = db_record.get('max_length', 512)
                        default_stride = db_record.get('stride', 50)
                        max_length_limit = default_max_length
                    else:
                        default_max_length = 512
                        default_stride = 50
                        max_length_limit = 1024

                    col1, col2 = st.columns(2)

                    with col1:
                        chunk_max_length = st.number_input(
                            "**MAX LENGTH**", min_value = 128, max_value = max_length_limit, value = default_max_length, key = "chunk_max_length"
                        )
                    with col2:
                        chunk_stride = st.number_input(
                            "**STRIDE**", min_value = 10, max_value = 100, value = default_stride, key = "chunk_stride"
                        )

        if st.button("**I N F E R E N C E**", type = "primary", width = "stretch", disabled = not model_exists):

            progress_bar = st.progress(0)
            
            try:

                model_name = st.session_state.get('ft_model_name_inference', 'google/gemma-2-2b')
                hf_token = st.session_state.get('ft_hf_token') or os.getenv('HF_TOKEN')
                model_type = st.session_state.get("inference_model_type", "GENERATIVE")

                inference = FineTuningInference(model_name, hf_token, model_type = model_type)

                with st.spinner("**LOADING MODEL ...**"):
                    inference.load_model(model_path, is_merged_model = True)
                    progress_bar.progress(0.25)

                with st.spinner("**RUNNING INFERENCE ...**"):
                    results_df = inference.predict_patents(
                        df, model_path,
                        selected_cols = selected_cols,
                        max_length = chunk_max_length,
                        stride = chunk_stride
                    )
                    progress_bar.progress(0.75)
                    
                progress_bar.progress(1.0)

                time.sleep(1.0)

                st.toast("**INFERENCE IS COMPLETE**")

                display_results = []
                full_results = []

                available_cols = DataProcessor.get_available_columns(df)

                for _, row in results_df.iterrows():

                    patent_id = row['출원_번호']

                    original_data = df[df['출원번호'] == patent_id].iloc[0] if not df[df['출원번호'] == patent_id].empty else None
                    
                    if original_data is not None:

                        display_results.append({
                            '출원 번호': patent_id,
                            '발명의 명칭': original_data.get('발명의 명칭', ''),
                            '분류 코드': row['분류_코드']
                        })

                        full_row = {}

                        for col in available_cols:
                            full_row[col] = original_data.get(col, '')
                        
                        full_row['분류 코드'] = row['분류_코드']
                        
                        full_results.append(full_row)

                results_df = pd.DataFrame(display_results)
                full_results_df = pd.DataFrame(full_results)
                
                st.session_state.inference_results = results_df
                st.session_state.inference_results_full = full_results_df

            except Exception as e:
                progress_bar.empty()
                st.code(str(e))

    if hasattr(st.session_state, 'inference_results') and st.session_state.inference_results is not None:

        st.markdown("---")
        
        st.markdown(
            "<h3 style = 'text-decoration: underline; text-decoration-thickness: 3px; text-underline-offset: 7px; text-decoration-color: slategray; margin-bottom: 20px;'>INFERENCE RESULT</h3>",
            unsafe_allow_html = True
        )

        results_df = st.session_state.inference_results
        full_results_df = st.session_state.inference_results_full

        with st.container():
            col_overview1, col_overview2 = st.columns(2)
            for col, title, value, color in zip(
                [col_overview1, col_overview2],
                ["TOTAL CLASSIFICATIONS", "COUNT OF UNIQUE CATEGORIES"],
                [len(results_df), results_df['분류 코드'].nunique()],
                ["#084298", "#084298"]
            ):
                with col:
                    st.markdown(
                        f"""
                        <div style ="
                            background: #dbeafe;
                            border-left: 7px solid {color};
                            padding: 15px;
                            border-radius: 10px;
                            text-align: center;
                            margin-bottom: 30px;
                        ">
                            <div style = "font-size: 17px; color: {color}; font-weight: 750;">{title}</div>
                            <div style = "font-size: 25px; font-weight: 700; color: dimgray;">{value}</div>
                        </div>
                        """,
                        unsafe_allow_html = True
                    )

        classification_groups = results_df.groupby('분류 코드')
        full_classification_groups = full_results_df.groupby('분류 코드')
        
        for category, group in classification_groups:
            with st.expander(f"**{category} ({len(group)}건)**", expanded = True):
                display_df = group[['출원 번호', '발명의 명칭', '분류 코드']].copy()
                st.dataframe(display_df, width = "stretch")

        st.markdown("<div style = 'height: 25px;'></div>", unsafe_allow_html = True)

        st.markdown(
            "<h3 style = 'text-decoration: underline; text-decoration-thickness: 3px; text-underline-offset: 7px; text-decoration-color: slategray;'>PREDICTION DISTRIBUTION</h3>",
            unsafe_allow_html = True
        )

        st.markdown("<div style = 'height: 45px;'></div>", unsafe_allow_html = True)

        classification_counts = results_df['분류 코드'].value_counts()
        
        st.bar_chart(classification_counts)

        st.markdown("---")

        excel_download.show_unified(full_results_df, full_classification_groups)