# tab/hpo.py

from dotenv import load_dotenv
import streamlit as st
from util.database.hpo_database import HPODatabase
import os
from component.model_list import show_model_list
from util.model_detector import detect_model_type
from util.optuna_tuner import OptunaHyperparameterTuner
from util.data_processor import DataProcessor
from util.text_chunker import SlidingWindowChunker
import time
import pandas as pd

load_dotenv()

def show():

    st.markdown(
        "<h3 style = 'text-decoration: underline; text-decoration-thickness: 3px; text-underline-offset: 7px; text-decoration-color: slategray; margin-bottom: 20px;'>ALL TIME RECORD</h3>",
        unsafe_allow_html = True
    )

    hpo_db = HPODatabase()

    best_records = hpo_db.get_best_record_by_model()

    best_records = sorted(
        best_records,
        key = lambda x : x.get("model_name", "").lower()
    )
        
    if best_records:
        for record in best_records:
            
            model_name = record['model_name']
            model_type = record['model_type']
            best_results = record['best_results']
            best_hyperparams = record['best_hyperparams']
            
            with st.expander(f"**{model_name}&nbsp;&nbsp;|&nbsp;&nbsp;{model_type}&nbsp;&nbsp;|&nbsp;&nbsp;(ACCURACY) {best_results.get('accuracy', 0) * 100:.1f}%**", expanded = False):

                with st.expander("**CLASSIFICATION REPORT**", expanded = False):
                
                    col1, col2, col3, col4 = st.columns(4)
                    
                    with col1:
                        st.metric("**ACCURACY**", f"{best_results.get('accuracy', 0):.2f}")
                    with col2:
                        st.metric("**F1 SCORE**", f"{best_results.get('f1', 0):.2f}")
                    with col3:
                        st.metric("**PRECISION**", f"{best_results.get('precision', 0):.2f}")
                    with col4:
                        st.metric("**RECALL**", f"{best_results.get('recall', 0):.2f}")

                with st.expander("**HYPERPARAMETER**", expanded = False):

                    hyperparams_cols = st.columns(3)
                    param_items = list(best_hyperparams.items())
                    
                    for i, (param, value) in enumerate(param_items):
                        col_idx = i % 3
                        with hyperparams_cols[col_idx]:
                            if isinstance(value, float):
                                if value < 0.001:
                                    st.write(f"**[ {param.upper()} ]**<br>{value:.5f}", unsafe_allow_html = True)
                                else:
                                    st.write(f"**[ {param.upper()} ]**<br>{value:.3f}", unsafe_allow_html = True)
                            else:
                                st.write(f"**[ {param.upper()} ]**<br>{value}", unsafe_allow_html = True)
                
                with st.expander("**ADDITION**", expanded = False):

                    col_info1, col_info2, col_info3 = st.columns(3)
                    
                    with col_info1:
                        st.write(f"**[ DATE ]**<br>{record['timestamp']}", unsafe_allow_html = True)
                    with col_info2:
                        st.write(f"**[ TRIALS ]**<br>{record['n_trials']}", unsafe_allow_html = True)
                    with col_info3:
                        st.write(f"**[ DATA COUNT ]**<br>{record['data_count']:,}", unsafe_allow_html = True)

    st.markdown("---")

    df = st.session_state.uploaded_df

    with st.expander("**COLUMNS TO USE FOR HPO**", expanded = True):

        col1, col2 = st.columns(2)

        with col1:
            label_options = ["-- SELECT --"] + df.columns.tolist()
            if "사용자태그" in df.columns:
                default_index = label_options.index("사용자태그")
            else:
                default_index = 0
            label_col = st.selectbox(
                "**LABEL COLUMN**",
                options = label_options,
                index = default_index,
                key = "hpo_label_col"
            )
            if label_col and label_col != "-- SELECT --":
                unique_labels = df[label_col].unique()
                with st.expander("**LABELS FOR THE DATA**", expanded = True):
                    st.write(sorted(unique_labels))

        with col2:
            selected_cols = st.multiselect(
                "**TEXT COLUMNS**",
                options = df.columns.tolist(),
                default = [col for col in ["발명의 명칭", "요약", "전체청구항"] if col in df.columns],
                key = "hpo_cols"
            )

    if not label_col or label_col == "-- SELECT --":
        st.info("🔴 PLEASE SELECT A LABEL COLUMN.")

    elif not selected_cols:
        st.info("🔴 PLEASE SELECT AT LEAST ONE TEXT COLUMN.")

    else:

        with st.expander("**SETTING**", expanded = True):

            with st.expander("**TRANSFORMERS**", expanded = True):

                col1, col2 = st.columns(2)
                
                with col1:
                    hf_token = st.text_input(
                        "**HUGGING FACE TOKEN**", 
                        value = st.session_state.get('hpo_hf_token') or os.getenv('HF_TOKEN'), 
                        type = "password", 
                        key = "hpo_hf_token"
                    )
                    
                with col2:

                    def on_model_name_change():
                        
                        new_model = st.session_state.get('hpo_model_name', '')
                        old_model = st.session_state.get('prev_model_name_for_detection', '')
                        
                        if new_model != old_model:
                            detected = detect_model_type(new_model)
                            st.session_state["hpo_model_type"] = detected
                            st.session_state["prev_model_name_for_detection"] = new_model
                            st.session_state["manual_model_type_override"] = False

                    model_name = st.text_input(
                        "**MODEL NAME**", 
                        value = st.session_state.get('hpo_model_name', 'google/gemma-2-2b'), 
                        key = "hpo_model_name",
                        on_change = on_model_name_change
                    )

                col3, col4 = st.columns(2)

                with col3:
                    show_model_list(key_prefix = "hpo")

                with col4:

                    col_a, col_b = st.columns(2)

                    if model_name and model_name.strip():

                        if "hpo_model_type" not in st.session_state:
                            detected_type = detect_model_type(model_name)
                            st.session_state["hpo_model_type"] = detected_type
                            st.session_state["prev_model_name_for_detection"] = model_name

                        with col_a:
                            rep_clicked = st.button(
                                "☑️ REPRESENTATION" if st.session_state.get("hpo_model_type") == "REPRESENTATION" else "🟪 REPRESENTATION",
                                width = 'stretch', key = "hpo_btn_rep"
                            )
                            if rep_clicked:
                                st.session_state["hpo_model_type"] = "REPRESENTATION"
                                st.session_state["manual_model_type_override"] = True
                                st.rerun()

                        with col_b:
                            gen_clicked = st.button(
                                "☑️ GENERATIVE" if st.session_state.get("hpo_model_type") == "GENERATIVE" else "🟪 GENERATIVE",
                                width = 'stretch', key = "hpo_btn_gen"
                            )
                            if gen_clicked:
                                st.session_state["hpo_model_type"] = "GENERATIVE"
                                st.session_state["manual_model_type_override"] = True
                                st.rerun()

                        model_type = st.session_state.get("hpo_model_type")

                    else:
                        model_type = "GENERATIVE"

            if model_name and model_name.strip():

                with st.expander("**HPO**", expanded = True):

                    col1, col2 = st.columns(2)

                    with col1:
                        n_trials = st.number_input(
                            "**NUMBER OF TRIALS**",
                            min_value = 1,
                            max_value = 100,
                            value = 5,
                            key = "hpo_n_trials"
                        )

                    with col2:
                        output_dir = st.text_input(
                            "**OUTPUT DIR**",
                            value = st.session_state.get("hpo_output_dir", r"C:\company\wips\hpo_"),
                            key = "hpo_output_dir"
                        )

        if not model_name or not model_name.strip():
            st.info("🔴 PLEASE ENTER A MODEL NAME TO PROCEED.")
            return

        output_dir_has_content = output_dir.strip() != ""

        if not output_dir_has_content:
            st.info("🔴 PLEASE ENTER A VALID OUTPUT PATH TO PROCEED.")
            return

        with st.expander("**SEARCH SPACE**", expanded = True):

            if model_type == "GENERATIVE":
                hyperparameter_options = ["LEARNING RATE", "BATCH SIZE", "GRADIENT ACCUMULATION STEPS", "EPOCHS", "LoRA RANK (R)", "LoRA ALPHA", "LoRA DROPOUT", "WARMUP STEPS"]
            else:
                hyperparameter_options = ["LEARNING RATE", "BATCH SIZE", "GRADIENT ACCUMULATION STEPS", "EPOCHS", "WARMUP STEPS", "WEIGHT DECAY"]
            
            selected_hyperparams = st.multiselect(
                "**HYPERPARAMETERS**",
                options = hyperparameter_options,
                default = ["LEARNING RATE", "EPOCHS", "WARMUP STEPS"],
                key = "hpo_selected_hyperparams"
            )

            if "LEARNING RATE" in selected_hyperparams:
                with st.expander("**LEARNING RATE**", expanded = False):
                    col1, col2 = st.columns(2)
                    with col1:
                        lr_low = st.number_input("**MIN**", value = 1e-5, format = "%.0e", key = "hpo_lr_low")
                    with col2:
                        lr_high = st.number_input("**MAX**", value = 5e-5, format = "%.0e", key = "hpo_lr_high")
            else:
                lr_low, lr_high = None, None

            if "BATCH SIZE" in selected_hyperparams:
                with st.expander("**BATCH SIZE**", expanded = False):
                    batch_choices = st.multiselect(
                        "**TRAIN / EVAL**", 
                        options = [1, 2, 4, 8, 16], 
                        default = [1, 2], 
                        key = "hpo_batch_choices"
                    )
            else:
                batch_choices = None

            if "GRADIENT ACCUMULATION STEPS" in selected_hyperparams:
                with st.expander("**GRADIENT ACCUMULATION STEPS**", expanded = False):
                    gas_choices = st.multiselect(
                        "**STEPS**", 
                        options = [1, 2, 4, 8, 16, 32], 
                        default = [1, 2], 
                        key = "hpo_gas_choices"
                    )
            else:
                gas_choices = None

            if "EPOCHS" in selected_hyperparams:
                with st.expander("**EPOCHS**", expanded = False):
                    col1, col2 = st.columns(2)
                    with col1:
                        epochs_low = st.number_input("**MIN**", value = 3, key = "hpo_epochs_low")
                    with col2:
                        epochs_high = st.number_input("**MAX**", value = 5, key = "hpo_epochs_high")
            else:
                epochs_low, epochs_high = None, None

            if "LoRA RANK (R)" in selected_hyperparams:
                with st.expander("**LoRA RANK (R)**", expanded = False):
                    lora_r_choices = st.multiselect(
                        "**RANK (R)**", 
                        options = [8, 16, 32, 64, 128, 256], 
                        default = [16, 32, 64], 
                        key = "hpo_lora_r_choices"
                    )
            else:
                lora_r_choices = None

            if "LoRA ALPHA" in selected_hyperparams:
                with st.expander("**LoRA ALPHA**", expanded = False):
                    lora_alpha_choices = st.multiselect(
                        "**ALPHA**", 
                        options = [16, 32, 64, 128, 256, 512], 
                        default = [32, 64, 128], 
                        key = "hpo_lora_alpha_choices"
                    )
            else:
                lora_alpha_choices = None

            if "LoRA DROPOUT" in selected_hyperparams:
                with st.expander("**LoRA DROPOUT**", expanded = False):
                    col1, col2 = st.columns(2)
                    with col1:
                        lora_dropout_low = st.number_input("**MIN**", value = 0.05, step = 0.01, key = "hpo_lora_dropout_low")
                    with col2:
                        lora_dropout_high = st.number_input("**MAX**", value = 0.25, step = 0.01, key = "hpo_lora_dropout_high")
            else:
                lora_dropout_low, lora_dropout_high = None, None

            if "WARMUP STEPS" in selected_hyperparams:
                with st.expander("**WARMUP STEPS**", expanded = False):
                    col1, col2 = st.columns(2)
                    with col1:
                        warmup_low = st.number_input("**MIN**", value = 10, key = "hpo_warmup_low")
                    with col2:
                        warmup_high = st.number_input("**MAX**", value = 50, key = "hpo_warmup_high")
            else:
                warmup_low, warmup_high = None, None

            if "WEIGHT DECAY" in selected_hyperparams:
                with st.expander("**WEIGHT DECAY**", expanded = False):
                    col1, col2 = st.columns(2)
                    with col1:
                        wd_low = st.number_input("**MIN**", value = 0.0, step = 0.001, key = "hpo_wd_low")
                    with col2:
                        wd_high = st.number_input("**MAX**", value = 0.1, step = 0.001, key = "hpo_wd_high")
            else:
                wd_low, wd_high = None, None

        search_space = {}
        
        if "LEARNING RATE" in selected_hyperparams and lr_low and lr_high:
            search_space['learning_rate'] = {'low' : lr_low, 'high' : lr_high, 'log' : True}
        if "BATCH SIZE" in selected_hyperparams and batch_choices:
            search_space['batch_size'] = {'choices' : batch_choices}
        if "GRADIENT ACCUMULATION STEPS" in selected_hyperparams and gas_choices:
            search_space['gradient_accumulation_steps'] = {'choices' : gas_choices}
        if "EPOCHS" in selected_hyperparams and epochs_low and epochs_high:
            search_space['epochs'] = {'low' : epochs_low, 'high' : epochs_high}
        if "LoRA RANK (R)" in selected_hyperparams and lora_r_choices:
            search_space['lora_r'] = {'choices' : lora_r_choices}
        if "LoRA ALPHA" in selected_hyperparams and lora_alpha_choices:
            search_space['lora_alpha'] = {'choices' : lora_alpha_choices}
        if "LoRA DROPOUT" in selected_hyperparams and lora_dropout_low is not None and lora_dropout_high is not None:
            search_space['lora_dropout'] = {'low' : lora_dropout_low, 'high' : lora_dropout_high}
        if "WARMUP STEPS" in selected_hyperparams and warmup_low is not None and warmup_high is not None:
            search_space['warmup_steps'] = {'low' : warmup_low, 'high' : warmup_high}
        if "WEIGHT DECAY" in selected_hyperparams and wd_low is not None and wd_high is not None:
            search_space['weight_decay'] = {'low' : wd_low, 'high' : wd_high}

        if not search_space:
            st.info("🔴 PLEASE SELECT AT LEAST ONE HYPERPARAMETER TO OPTIMIZE.")

        if st.button("**O P T I M I Z A T I O N**", type = "primary", width = "stretch", disabled = not search_space):
            
            progress_bar = st.progress(0)

            try:

                tuner = OptunaHyperparameterTuner(model_name, hf_token, model_type = model_type)

                with st.spinner("**INITIALIZING TOKENIZER ...**"):              
                    tuner.initialize_tokenizer()
                    progress_bar.progress(0.15)

                with st.spinner("**PREPROCESSING DATA ...**"):
                    processed_df = DataProcessor.prepare_data(tuner, df, selected_cols = selected_cols, label_col = label_col)
                    chunker = SlidingWindowChunker(tuner.tokenizer)
                    df_chunked = chunker.create_chunked_dataset(processed_df, max_length = 512, stride = 50)                   
                    tokenized_dataset, test_df = DataProcessor.create_balanced_datasetdict(
                        df_chunked, tuner.tokenizer, test_size = 0.2
                    )                   
                    tuner.set_data(tokenized_dataset, test_df, tuner.labels_list, tuner.label2id, tuner.id2label)
                    progress_bar.progress(0.30)

                with st.spinner("**CREATING OPTIMIZATION STUDY ...**"):
                    study_name = f"hpo_{int(time.time())}"
                    tuner.create_study(study_name = study_name)
                    progress_bar.progress(0.45)

                with st.spinner(f"**RUNNING OPTIMIZATION ( {n_trials} TRIALS ) ...**"):
                    study = tuner.optimize(output_dir, search_space, n_trials = n_trials)
                    progress_bar.progress(0.90)

                with st.spinner("**SAVING RESULT ...**"):

                    results_path = os.path.join(output_dir, "hpo_results.json")
                    tuner.save_study_results(results_path)

                    hpo_db = HPODatabase()

                    all_trials = []

                    for trial in study.trials:
                        trial_info = {
                            'number' : trial.number,
                            'value' : trial.value,
                            'params' : trial.params,
                            'state' : trial.state.name,
                            'datetime_start' : trial.datetime_start.isoformat() if trial.datetime_start else None,
                            'datetime_complete' : trial.datetime_complete.isoformat() if trial.datetime_complete else None,
                            'user_attrs' : trial.user_attrs if hasattr(trial, 'user_attrs') else {}
                        }
                        all_trials.append(trial_info)

                    best_metrics = tuner.get_best_trial_metrics()
                    best_results = best_metrics if best_metrics else {
                        'accuracy': None,
                        'f1': None,
                        'precision': None,
                        'recall': None
                    }

                    hpo_record_id = hpo_db.save_hpo_record(
                        model_name = model_name,
                        data_count = len(processed_df),
                        labels_list = tuner.labels_list,
                        text_columns = selected_cols,
                        n_trials = n_trials,
                        best_hyperparams = tuner.get_best_params(),
                        best_results = best_results,
                        all_trials = all_trials,
                        study_name = study_name,
                        model_type = model_type
                    )

                    hpo_db.cleanup_old_records()

                    progress_bar.progress(1.0)

                time.sleep(1.0)

                st.toast("**HYPERPARAMETER OPTIMIZATION IS COMPLETE**")

                st.markdown("---")

                st.markdown(
                    "<h3 style = 'text-decoration: underline; text-decoration-thickness: 3px; text-underline-offset: 7px; text-decoration-color: slategray; margin-bottom: 20px;'>HPO RESULT</h3>",
                    unsafe_allow_html = True
                )

                best_params = tuner.get_best_params()
                best_value = tuner.get_best_value()

                with st.container():
                    col_overview1, col_overview2, col_overview3 = st.columns(3)
                    for col, title, value, color in zip(
                        [col_overview1, col_overview2, col_overview3],
                        ["TOTAL TRIALS", "(COMPLETE) TRIALS", "PEAK ACCURACY"],
                        [len(study.trials), len([t for t in study.trials if t.state.name == 'COMPLETE']), f"{best_value:.4f}"],
                        ["#084298", "#084298", "#084298"]
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

                with st.expander("**OPTIMAL HYPERPARAMETER**", expanded = True):
                    best_params_df = pd.DataFrame([
                        {"PARAMETER" : k, "VALUE" : v} for k, v in best_params.items()
                    ])
                    st.dataframe(best_params_df, width = 'stretch')

                if len(study.trials) > 1:
                    with st.expander("**PLOT**", expanded = True):
                        plots = tuner.create_optimization_plots()
                        for fig in plots:
                            st.plotly_chart(fig, width = 'stretch')

                with st.expander("**TRIAL LOG**", expanded = False):

                    trials_data = []

                    for i, trial in enumerate(study.trials):
                        trial_info = {
                            "TRIAL" : trial.number,
                            "VALUE" : trial.value if trial.value else 0.0,
                            "STATE" : trial.state.name,
                            "DURATION" : str(trial.duration) if trial.duration else "N/A"
                        }
                        for param, value in trial.params.items():
                            trial_info[param] = value
                        trials_data.append(trial_info)
                    
                    trials_df = pd.DataFrame(trials_data)
                    st.dataframe(trials_df, width = 'stretch')

                st.session_state.hpo_results = {
                    'study' : study,
                    'best_params' : best_params,
                    'best_value' : best_value,
                    'output_dir' : output_dir,
                    'hpo_record_id' : hpo_record_id
                }

            except Exception as e:
                progress_bar.empty()
                st.code(str(e))