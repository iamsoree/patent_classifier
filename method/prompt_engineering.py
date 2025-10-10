# method/prompt_engineering.py

import streamlit as st
from component.prompt_engineering import category_setting, prompt_setting
from util import lm_studio
from util import excel_download

def show():
    
    if st.session_state.uploaded_df is not None:

        st.markdown(
            '''
            <h2 style = "
                border: 3px dashed slategray;
                display: inline-block;
                padding: 5px 7px;
                margin: 5px 0 30px 0;
                border-radius: 5px;
                font-weight: bold;
            ">
                PROMPT ENGINEERING
            </h2>
            ''',
            unsafe_allow_html = True
        )

        df = st.session_state.uploaded_df

        with st.expander("**COLUMNS TO INCLUDE IN PROMPT**", expanded = True):
        
            col1, col2 = st.columns([2, 1])
            
            with col1:
                selected_columns = st.multiselect(
                    "**SELECTED COLUMNS**",
                    options = df.columns.tolist(),
                    default = [col for col in ["발명의 명칭", "요약", "전체청구항"] if col in df.columns],
                    help = "The contents of the selected columns are combined and sent to the prompt."
                )
                
            with col2:
                if len(selected_columns) > 1:
                    combine_method = st.selectbox(
                        "**COLUMN MERGE SCHEME**",
                        ["SPACE", "LINE BREAKS", "CUSTOM DELIMITER"]
                    )
                    if combine_method == "CUSTOM DELIMITER":
                        custom_separator = st.text_input("DELIMITER", value = " | ")
                    else:
                        custom_separator = " " if combine_method == "SPACE" else "\n"
                else:
                    custom_separator = ""
        
        if selected_columns:
    
            category_setting.show()
    
            prompt_has_content = prompt_setting.show(selected_columns, df, custom_separator)

            if prompt_has_content:
                lm_studio.settings()
                api_connection_success = st.session_state.get('api_connection_success', False)
                if not api_connection_success:
                    st.info("🔴 PLEASE COMPLETE API CONNECTION FIRST.")
                if st.button("**C L A S S I F Y**", type = "primary", width = "stretch", disabled = not api_connection_success):
                    if api_connection_success:
                        lm_studio.inference(selected_columns, df, custom_separator, st.session_state.step1_prompt, st.session_state.step2_prompt, st.session_state.categories)

            else:
                st.info("🔴 PLEASE WRITE THE PROMPT TEMPLATE FIRST.")

        else:
            st.info("🔴 PLEASE SELECT AT LEAST ONE COLUMN.")

    else:
        st.info("⬅️ 특허 문서를 업로드해 주세요.")
    
    if hasattr(st.session_state, 'classification_results') and st.session_state.classification_results is not None:

        st.markdown("---")

        st.markdown(
            "<h3 style = 'text-decoration: underline; text-decoration-thickness: 3px; text-underline-offset: 7px; text-decoration-color: slategray; margin-bottom: 20px;'>CLASSIFICATION RESULT</h3>",
            unsafe_allow_html = True
        )

        results_df = st.session_state.classification_results
        full_results_df = st.session_state.classification_results_full

        with st.container():
            col_overview1, col_overview2, col_overview3 = st.columns(3)
            for col, title, value, color in zip(
                [col_overview1, col_overview2, col_overview3],
                ["TOTAL CLASSIFICATIONS", "COUNT OF UNIQUE CATEGORIES", "NUMBER OF ERRORS"],
                [len(results_df), results_df[results_df['분류 코드'] != "ERROR"]['분류 코드'].nunique(), len(results_df[results_df['분류 코드'] == "ERROR"])],
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