# tab/train.py

from dotenv import load_dotenv
import streamlit as st
from util.data_processor import DataProcessor
import os
from component.model_list import show_model_list
from util.model_detector import detect_model_type
from util.finetuning_train import FineTuningTrain
from util.text_chunker import SlidingWindowChunker
import time
from util.database.ft_database import TrainingDatabase

load_dotenv()

def show():

    df = st.session_state.uploaded_df

    with st.expander("**COLUMNS TO USE FOR TRAIN**", expanded = True):

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
                key = "train_label_col"
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
                key = "train_cols"
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
                        value = st.session_state.get('ft_hf_token') or os.getenv('HF_TOKEN'), 
                        type = "password", 
                        key = "ft_hf_token_train"
                    )

                with col2:

                    def on_model_name_change():

                        new_model = st.session_state.get('ft_model_name_train', '')
                        old_model = st.session_state.get('prev_model_name_for_detection', '')
                        
                        if new_model != old_model:
                            detected = detect_model_type(new_model)
                            st.session_state["train_model_type"] = detected
                            st.session_state["prev_model_name_for_detection"] = new_model
                            st.session_state["manual_model_type_override"] = False

                    model_name = st.text_input(
                        "**MODEL NAME**", 
                        value = st.session_state.get('ft_model_name_train', 'google/gemma-2-2b'), 
                        key = "ft_model_name_train",
                        on_change = on_model_name_change
                    )

                col3, col4 = st.columns(2)

                with col3:
                    show_model_list(key_prefix = "train")

                with col4:

                    col_a, col_b = st.columns(2)

                    if model_name and model_name.strip():

                        if "train_model_type" not in st.session_state:
                            detected_type = detect_model_type(model_name)
                            st.session_state["train_model_type"] = detected_type
                            st.session_state["prev_model_name_for_detection"] = model_name

                        with col_a:
                            rep_clicked = st.button(
                                "☑️ REPRESENTATION" if st.session_state.get("train_model_type") == "REPRESENTATION" else "🟪 REPRESENTATION",
                                width = 'stretch', key = "train_btn_rep"
                            )
                            if rep_clicked:
                                st.session_state["train_model_type"] = "REPRESENTATION"
                                st.session_state["manual_model_type_override"] = True
                                st.rerun()

                        with col_b:
                            gen_clicked = st.button(
                                "☑️ GENERATIVE" if st.session_state.get("train_model_type") == "GENERATIVE" else "🟪 GENERATIVE",
                                width = 'stretch', key = "train_btn_gen"
                            )
                            if gen_clicked:
                                st.session_state["train_model_type"] = "GENERATIVE"
                                st.session_state["manual_model_type_override"] = True
                                st.rerun()

                        model_type = st.session_state.get("train_model_type")

                    else:
                        model_type = "GENERATIVE"

            if model_name and model_name.strip():

                with st.expander("**HYPERPARAMETER**", expanded = True):

                    if model_type == "GENERATIVE":

                        with st.expander("**QUANTIZATION**", expanded = False):
                            col1, col2, col3, col4 = st.columns(4)
                            with col1:
                                load_in_4bit = st.checkbox("**4 - BIT QUANTIZATION**", value = True)
                            with col2:
                                bnb_4bit_quant_type = st.selectbox("**QUANTIZATION TYPE**", ["nf4", "fp4"], index = 0)
                            with col3:
                                bnb_4bit_compute_dtype = st.selectbox("**COMPUTE DTYPE**", ["float16", "bfloat16", "float32"], index = 0)
                            with col4:
                                bnb_4bit_use_double_quant = st.checkbox("**DOUBLE QUANTIZATION**", value = True)

                        with st.expander("**LoRA**", expanded = False):
                            col1, col2, col3 = st.columns(3)
                            with col1:
                                lora_r = st.number_input("**LoRA RANK (R)**", min_value = 4, max_value = 256, value = 64)
                                lora_alpha = st.number_input("**LoRA ALPHA**", min_value = 8, max_value = 512, value = 128)
                            with col2:
                                lora_dropout = st.number_input("**LoRA DROPOUT**", min_value = 0.0, max_value = 0.5, value = 0.1, step = 0.01)
                                bias_setting = st.selectbox("**BIAS**", ["none", "all", "lora_only"], index = 0)
                            with col3:
                                task_type = st.selectbox("**TASK TYPE**", ["SEQ_CLS", "CAUSAL_LM", "SEQ_2_SEQ_LM"], index = 0)
                                target_modules = st.multiselect(
                                    "**TARGET MODULES**",
                                    options = ['k_proj', 'gate_proj', 'v_proj', 'up_proj', 'q_proj', 'o_proj', 'down_proj'],
                                    default = ['k_proj', 'gate_proj', 'v_proj', 'up_proj', 'q_proj', 'o_proj', 'down_proj']
                                )

                    elif model_type == "REPRESENTATION":
                        load_in_4bit = False
                        bnb_4bit_quant_type = None
                        bnb_4bit_compute_dtype = None
                        bnb_4bit_use_double_quant = False
                        lora_r = None
                        lora_alpha = None
                        lora_dropout = None
                        bias_setting = None
                        task_type = None
                        target_modules = None

                    with st.expander("**SFT**", expanded = False):

                        col1, col2, col3 = st.columns(3)

                        default_lr = 2e-5 if model_type == "GENERATIVE" else 5e-5
                        default_train_batch = 2 if model_type == "GENERATIVE" else 4
                        default_eval_batch = 2 if model_type == "GENERATIVE" else 4
                        default_grad_accum = 2 if model_type == "GENERATIVE" else 4

                        with col1:
                            epochs = st.number_input("**EPOCHS**", min_value = 1, max_value = 25, value = 5)
                            learning_rate = st.number_input("**LEARNING RATE**", min_value = 1e-7, max_value = 1e-3, value = default_lr, format = "%.0e")
                            per_device_train_batch_size = st.number_input("**TRAIN BATCH SIZE**", min_value = 1, max_value = 16, value = default_train_batch)
                            per_device_eval_batch_size = st.number_input("**EVAL BATCH SIZE**", min_value = 1, max_value = 16, value = default_eval_batch)
                        with col2:
                            warmup_steps = st.number_input("**WARMUP STEPS**", min_value = 0, max_value = 100, value = 50)
                            test_size = st.number_input("**TEST SIZE**", min_value = 0.1, max_value = 0.3, value = 0.2)
                            gradient_accumulation_steps = st.number_input("**GRADIENT ACCUMULATION STEPS**", min_value = 1, max_value = 32, value = default_grad_accum)
                            logging_steps = st.number_input("**LOGGING STEPS**", min_value = 1, max_value = 100, value = 10)
                        with col3:
                            max_length = st.number_input("**MAX LENGTH**", min_value = 128, max_value = 1024, value = 512)
                            stride = st.number_input("**STRIDE**", min_value = 10, max_value = 100, value = 50)
                            if model_type == "GENERATIVE":
                                optim = st.selectbox("**OPTIM**", ["paged_adamw_32bit", "adamw_torch", "adamw_hf", "sgd"], index = 0)
                                lr_scheduler_type = st.selectbox("**LR SCHEDULER TYPE**", ["cosine", "linear", "polynomial", "constant"], index = 0)
                            else:
                                weight_decay = st.number_input("**WEIGHT DECAY**", min_value = 0.0, max_value = 0.1, value = 0.01, step = 0.001)

                        col4, col5 = st.columns([1, 2])

                        with col4:
                            gradient_checkpointing = st.checkbox("**GRADIENT CHECKPOINTING**", value = True)
                        with col5:
                            fp16 = st.checkbox("**FP16**", value = True)

        if not model_name or not model_name.strip():
            st.info("🔴 PLEASE ENTER A MODEL NAME TO PROCEED.")
            return
                
        with st.expander("**OUTPUT DIR**", expanded = True):

            path = st.text_input(
                "**PATH**",
                value = st.session_state.get("path", r"C:\company\wips"),
                key = "path"
            )

            folder = st.text_input(
                "**FOLDER**",
                value = st.session_state.get("folder", "ft_"),
                key = "folder"
            )

        if path and folder:
            output_dir = os.path.join(path.strip(), folder.strip())
            st.session_state.output_dir = output_dir
        else:
            output_dir = ""

        output_dir_has_content = output_dir.strip() != ""

        if not output_dir_has_content:
            st.info("🔴 PLEASE ENTER A VALID OUTPUT PATH TO PROCEED.")

        if st.button("**T R A I N**", type = "primary", width = "stretch", disabled = not output_dir_has_content):

            progress_bar = st.progress(0)

            try:

                model_name = st.session_state.get('ft_model_name_train', 'google/gemma-2-2b')
                hf_token = st.session_state.get('ft_hf_token') or os.getenv('HF_TOKEN')

                trainer = FineTuningTrain(model_name, hf_token, model_type = model_type)

                with st.spinner("**INITIALIZING MODEL ...**"):
                    trainer.initialize_tokenizer()
                    progress_bar.progress(0.15)

                with st.spinner("**PREPROCESSING DATA ...**"):

                    processed_df = DataProcessor.prepare_data(trainer, df, selected_cols = selected_cols, label_col = label_col)
                    
                    chunker = SlidingWindowChunker(trainer.tokenizer)
                    
                    df_chunked = chunker.create_chunked_dataset(processed_df, max_length, stride)

                    tokenized_dataset, test_df = DataProcessor.create_balanced_datasetdict(
                        df_chunked, trainer.tokenizer, test_size = test_size
                    )

                    progress_bar.progress(0.30)

                with st.spinner("**CONFIGURING MODEL ...**"):

                    if model_type == "GENERATIVE":

                        bnb_config_params = {
                            'load_in_4bit' : load_in_4bit,
                            'bnb_4bit_quant_type' : bnb_4bit_quant_type,
                            'bnb_4bit_compute_dtype' : bnb_4bit_compute_dtype,
                            'bnb_4bit_use_double_quant' : bnb_4bit_use_double_quant
                        }

                        lora_config_params = {
                            'lora_alpha' : lora_alpha,
                            'lora_dropout' : lora_dropout,
                            'r' : lora_r,
                            'bias' : bias_setting,
                            'task_type' : task_type,
                            'target_modules' : target_modules
                        }

                    else:
                        bnb_config_params = None
                        lora_config_params = None

                    trainer.setup_model(bnb_config_params, lora_config_params)

                    progress_bar.progress(0.45)

                with st.spinner("**TRAINING MODEL ...**"):

                    training_config_params = {
                        'num_train_epochs' : epochs,
                        'learning_rate' : learning_rate,
                        'per_device_train_batch_size' : per_device_train_batch_size,
                        'per_device_eval_batch_size' : per_device_eval_batch_size,
                        'gradient_accumulation_steps' : gradient_accumulation_steps,
                        'warmup_steps' : warmup_steps,
                        'logging_steps' : logging_steps,
                        'fp16' : fp16,
                        'gradient_checkpointing' : gradient_checkpointing
                    }

                    if model_type == "REPRESENTATION":
                        training_config_params['weight_decay'] = weight_decay
                    else:
                        training_config_params['max_length'] = max_length
                        training_config_params['optim'] = optim
                        training_config_params['lr_scheduler_type'] = lr_scheduler_type

                    eval_results = trainer.train_model(
                        tokenized_dataset, test_df, output_dir,
                        bnb_config_params, lora_config_params, training_config_params
                    )

                    progress_bar.progress(0.90)

                with st.spinner("**SAVING MODEL ...**"):

                    trainer.save_model(output_dir)

                    progress_bar.progress(1.0)
                
                time.sleep(1.0)

                try:
                    test_save_path = os.path.join(output_dir, "test_data.csv")
                    test_df.to_csv(test_save_path, index = False, encoding = "utf-8-sig")
                    print(test_save_path)

                except Exception as e:
                    print(e)

                st.toast("**TRAIN IS COMPLETE**")

                st.markdown("---")
        
                st.markdown(
                    "<h3 style = 'text-decoration: underline; text-decoration-thickness: 3px; text-underline-offset: 7px; text-decoration-color: slategray; margin-bottom: 25px;'>TRAIN RESULT</h3>",
                    unsafe_allow_html = True
                )

                with st.container():
                    col1, col2, col3, col4 = st.columns(4)
                    for col, title, value, color in zip(
                        [col1, col2, col3, col4],
                        ["ACCURACY", "F1 SCORE", "PRECISION", "RECALL"],
                        [f"{eval_results['eval_accuracy']:.2f}", f"{eval_results['eval_f1']:.2f}", f"{eval_results['eval_precision']:.2f}", f"{eval_results['eval_recall']:.2f}"],
                        ["#084298", "#084298", "#084298", "#084298"]
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

                st.session_state.model_info = {
                    "model_path" : output_dir,
                    "labels_list" : trainer.labels_list,
                    "label2id" : trainer.label2id,
                    "id2label" : trainer.id2label
                }

                try:

                    db = TrainingDatabase()
                    
                    hyperparams = {
                        'model_type' : model_type,
                        'num_train_epochs' : epochs,
                        'learning_rate' : learning_rate,
                        'per_device_train_batch_size' : per_device_train_batch_size,
                        'per_device_eval_batch_size' : per_device_eval_batch_size,
                        'gradient_accumulation_steps' : gradient_accumulation_steps,
                        'warmup_steps' : warmup_steps,
                        'max_length' : max_length,
                        'stride' : stride,
                        'logging_steps' : logging_steps,
                        'fp16' : fp16,
                        'gradient_checkpointing' : gradient_checkpointing
                    }

                    if model_type == "GENERATIVE":
                        hyperparams.update({
                            'lora_r' : lora_r,
                            'lora_alpha' : lora_alpha,
                            'lora_dropout' : lora_dropout,
                            'bias' : bias_setting,
                            'task_type' : task_type,
                            'target_modules' : target_modules,
                            'load_in_4bit' : load_in_4bit,
                            'bnb_4bit_quant_type' : bnb_4bit_quant_type,
                            'bnb_4bit_compute_dtype' : bnb_4bit_compute_dtype,
                            'bnb_4bit_use_double_quant' : bnb_4bit_use_double_quant,
                            'optim' : optim,
                            'lr_scheduler_type' : lr_scheduler_type
                        })

                    else:
                        hyperparams['weight_decay'] = weight_decay

                    label_mappings = {
                        'label2id' : trainer.label2id,
                        'id2label' : trainer.id2label
                    }
                    
                    id = db.save_training_record(
                        model_name = model_name,
                        data_count = len(processed_df),
                        labels_list = trainer.labels_list,
                        text_columns = selected_cols,
                        hyperparameters = hyperparams,
                        output_path = output_dir,
                        results = eval_results,
                        label_mappings = label_mappings
                    )

                    st.info(f"🔵 ID : {id}")
                    
                    st.session_state.current_training_record = {
                        'id' : id,
                        'saved' : True
                    }

                except Exception as e:
                    st.info(f"🔴 {str(e)}")
                    st.session_state.current_training_record = {
                        'saved' : False,
                        'error' : str(e)
                    }

            except Exception as e:
                progress_bar.empty()
                st.code(str(e))