# component/model_list.py

import streamlit as st

def show_model_list(key_prefix = ""):
    
    model_info = {
        "google/gemma-3-12b" : {"PARADIGM" : "GENERATIVE", "PARAMS" : "12.2B", "ARCH" : "gemma", "TASKS" : "Image-Text-to-Text", "CONTEXT_LENGTH" : "131,072"},
        "qwen/qwen3-14b" : {"PARADIGM" : "GENERATIVE", "PARAMS" : "14.8B", "ARCH" : "qwen", "TASKS" : "Text Generation", "CONTEXT_LENGTH" : "32,768 (YaRN : 131,072)"},
        "openai/gpt-oss-20b" : {"PARADIGM" : "GENERATIVE", "PARAMS" : "21.5B", "ARCH" : "gpt-oss", "TASKS" : "Text Generation", "CONTEXT_LENGTH" : "131,072"},
        "google/gemma-2-2b" : {"PARADIGM" : "GENERATIVE", "PARAMS" : "2.61B", "ARCH" : "gemma", "TASKS" : "Text Generation", "CONTEXT_LENGTH" : "8,192"},
        "Qwen/Qwen3-1.7B" : {"PARADIGM" : "GENERATIVE", "PARAMS" : "2.03B", "ARCH" : "qwen", "TASKS" : "Text Generation", "CONTEXT_LENGTH" : "32,768"},
        "Qwen/Qwen3-0.6B" : {"PARADIGM" : "GENERATIVE", "PARAMS" : "0.752B", "ARCH" : "qwen", "TASKS" : "Text Generation", "CONTEXT_LENGTH" : "32,768"},
        "meta-llama/Llama-3.2-1B" : {"PARADIGM" : "GENERATIVE", "PARAMS" : "1.24B", "ARCH" : "llama", "TASKS" : "Text Generation", "CONTEXT_LENGTH" : "131,072"},
        "meta-llama/Llama-3.2-3B" : {"PARADIGM" : "GENERATIVE", "PARAMS" : "3.21B", "ARCH" : "llama", "TASKS" : "Text Generation", "CONTEXT_LENGTH" : "131,072"},
        "microsoft/phi-4" : {"PARADIGM" : "GENERATIVE", "PARAMS" : "14.7B", "ARCH" : "phi", "TASKS" : "Text Generation", "CONTEXT_LENGTH" : "16,384"},
        "google/gemma-3-1b" : {"PARADIGM" : "GENERATIVE", "PARAMS" : "1B", "ARCH" : "gemma", "TASKS" : "Text Generation", "CONTEXT_LENGTH" : "32,768"},
        "LiquidAI/LFM2-1.2B" : {"PARADIGM" : "GENERATIVE", "PARAMS" : "1.17B", "ARCH" : "lfm2", "TASKS" : "Text Generation", "CONTEXT_LENGTH" : "32,768"},
        "ibm/granite-3.2-8b" : {"PARADIGM" : "GENERATIVE", "PARAMS" : "8.17B", "ARCH" : "granite", "TASKS" : "Text Generation", "CONTEXT_LENGTH" : "131,072"},
        "allenai/scibert_scivocab_cased" : {"PARADIGM" : "REPRESENTATION", "PARAMS" : "0.11B", "ARCH" : "bert", "TASKS" : "Text Classification", "CONTEXT_LENGTH" : "512"},
        "microsoft/MiniLM-L12-H384-uncased" : {"PARADIGM" : "REPRESENTATION", "PARAMS" : "0.033B", "ARCH" : "minilm", "TASKS" : "Text Classification", "CONTEXT_LENGTH" : "512"},
        "google/electra-small-discriminator" : {"PARADIGM" : "REPRESENTATION", "PARAMS" : "0.014B", "ARCH" : "electra", "TASKS" : "Text Classification", "CONTEXT_LENGTH" : "512"}
    }
    
    def parse_params_to_float(params_str):
        if params_str.endswith('B'):
            return float(params_str.replace('B', ''))
    
    with st.popover("**MODEL LIST**", width = 'content'):

        st.markdown(
            """
            <div style = "margin-top: 15px;">
                <span style = "
                    background-color: #dbeafe;
                    color: dimgray;
                    padding: 3px 5px;
                    border-radius: 5px;
                    font-weight: bold;
                    font-size: 30px;
                ">
                MODEL LIST
                </span>
            </div>
            """,
            unsafe_allow_html = True
        )

        st.markdown("<br>", unsafe_allow_html = True)

        search_query = st.text_input(
            "**SEARCH**", 
            placeholder = "SEARCH BY MODEL NAME ...",
            key = f"{key_prefix}_model_search"
        )

        col1, col2, col3 = st.columns(3)

        with col1:
            paradigm_filter = st.selectbox(
                "**FILTER BY PARADIGM**", 
                ["ALL", "REPRESENTATION", "GENERATIVE"],
                key = f"{key_prefix}_paradigm_filter"
            )
        
        with col2:
            params_filter = st.selectbox(
                "**FILTER BY PARAMS**", 
                ["ALL", "~ 4B", "5B ~ 9B", "10B ~ 19B", "20B ~"],
                key = f"{key_prefix}_params_filter"
            )
        with col3:
            all_archs = ["ALL"] + sorted(list(set(info["ARCH"] for info in model_info.values())))
            arch_filter = st.selectbox(
                "**FILTER BY ARCH**", 
                all_archs,
                key = f"{key_prefix}_arch_filter"
            )
        
        sort_by = st.selectbox(
            "**SORT BY**", 
            ["ARCH (A - Z)", "ARCH (Z - A)", "PARAMS (LOW - HIGH)", "PARAMS (HIGH - LOW)"],
            index = 0,
            key = f"{key_prefix}_model_sort"
        )
        
        filtered_models = list(model_info.items())

        if search_query:
            filtered_models = [(k, v) for k, v in filtered_models if search_query.lower() in k.lower()]

        if paradigm_filter != "ALL":
            filtered_models = [(k, v) for k, v in filtered_models if v["PARADIGM"] == paradigm_filter]
        
        if params_filter != "ALL":

            def params_in_range(params_str, filter_type):

                params_val = parse_params_to_float(params_str)
                
                if filter_type == "~ 4B":
                    return params_val < 5.0
                elif filter_type == "5B ~ 9B":
                    return 5.0 <= params_val < 10.0
                elif filter_type == "10B ~ 19B":
                    return 10.0 <= params_val < 20.0
                elif filter_type == "20B ~":
                    return params_val >= 20.0
                
                return True
            
            filtered_models = [(k, v) for k, v in filtered_models if params_in_range(v["PARAMS"], params_filter)]

        if arch_filter != "ALL":
            filtered_models = [(k, v) for k, v in filtered_models if v["ARCH"] == arch_filter]

        if sort_by == "PARAMS (LOW - HIGH)":
            filtered_models = sorted(filtered_models, key = lambda x : parse_params_to_float(x[1]["PARAMS"]))
        elif sort_by == "PARAMS (HIGH - LOW)":
            filtered_models = sorted(filtered_models, key = lambda x : parse_params_to_float(x[1]["PARAMS"]), reverse = True)
        elif sort_by == "ARCH (A - Z)":
            filtered_models = sorted(filtered_models, key = lambda x : x[1]["ARCH"])
        elif sort_by == "ARCH (Z - A)":
            filtered_models = sorted(filtered_models, key = lambda x : x[1]["ARCH"], reverse = True)
        
        st.markdown("---")

        if not filtered_models:
            st.markdown(f"### *NO MODELS MATCH THE SELECTED FILTERS.*")

        else:

            st.markdown(f"### *FOUND {len(filtered_models)} MODEL(S) :*")
            
            for model, info in filtered_models:
                with st.container():

                    col1, col2 = st.columns([3, 1])
                    
                    with col1:
                        st.markdown(f"##### **{model}**")
                        st.caption(f"PARADIGM : {info['PARADIGM']} | PARAMS : {info['PARAMS']} | ARCH : {info['ARCH']} | TASKS : {info['TASKS']} | CONTEXT_LENGTH : {info['CONTEXT_LENGTH']}")
                    with col2:
                        if st.button("🟦", key = f"{key_prefix}_pick_{model}", help = model):
                            st.session_state.picked_model = model
                            st.rerun()
                    
                    st.divider()
        
        if hasattr(st.session_state, 'picked_model'):

            st.code(st.session_state.picked_model)

            st.markdown("<br>", unsafe_allow_html = True)
            
            if st.button("🔄 **CHOOSE ANOTHER**", key = f"{key_prefix}_choose_another", width = 'stretch'):
                del st.session_state.picked_model
                st.rerun()