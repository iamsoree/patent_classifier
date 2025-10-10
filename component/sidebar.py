# component/sidebar.py

import streamlit as st
import pandas as pd

def show():

    def sidebar_title(title : str):
        st.sidebar.markdown(
            f"""
            <span style = "
                background-color: #dbeafe;
                color: dimgray;
                padding: 3px 5px;
                border-radius: 5px;
                font-weight: bold;
                font-size: 23px;
            ">
            {title}
            </span>
            """,
            unsafe_allow_html = True
        )

    sidebar_title("HOW TO USE")

    with st.sidebar.expander("**PROMPT ENGINEERING**", expanded = False):
        st.markdown("""
        1. (SIDEBAR_DATA)
        2. DATA PREVIEW
        3. (SIDEBAR_APPROACH) PROMPT ENGINEERING
        4. COLUMNS TO INCLUDE IN PROMPT
        5. CATEGORY TO CLASSIFY
        6. PROMPT TEMPLATE
        7. LM STUDIO SETTING
        8. CLASSIFY
        """)

    with st.sidebar.expander("**FINE TUNING**", expanded = False):
        with st.expander("**INFERENCE**", expanded = False):
            st.markdown("""
            1. (SIDEBAR_DATA)
            2. DATA PREVIEW
            3. (SIDEBAR_APPROACH) FINE TUNING
            4. (TAB) INFERENCE
            5. COLUMNS TO USE FOR INFERENCE
            6. MODEL TO USE FOR INFERENCE
            7. SETTING
            8. INFERENCE
            """)
        with st.expander("**TRAIN**", expanded = False):
            st.markdown("""
            1. (SIDEBAR_DATA)
            2. DATA PREVIEW
            3. (SIDEBAR_APPROACH) FINE TUNING
            4. (TAB) TRAIN
            5. COLUMNS TO USE FOR TRAIN
            6. SETTING
            7. OUTPUT DIR
            8. TRAIN
            """)
        with st.expander("**HYPERPARAMETER OPTIMIZATION**", expanded = False):
            st.markdown("""
            1. (SIDEBAR_DATA)
            2. DATA PREVIEW
            3. (SIDEBAR_APPROACH) FINE TUNING
            4. (TAB) HYPERPARAMETER OPTIMIZATION
            5. ALL TIME RECORD
            6. COLUMNS TO USE FOR HPO
            7. SETTING
            8. SEARCH SPACE
            9. OPTIMIZATION
            """)
        with st.expander("**HISTORY**", expanded = False):
            st.markdown("""
            1. (SIDEBAR_DATA)
            2. DATA PREVIEW
            3. (SIDEBAR_APPROACH) FINE TUNING
            4. (TAB) HISTORY
            5. OVERALL SUMMARY
            6. ACCURACY BY MODEL
            7. PREVIOUS RECORD
            8. DOWNLOAD HISTORY
            """)

    st.sidebar.markdown("---")
        
    sidebar_title("DATA")

    uploaded_file = st.sidebar.file_uploader(
        "**UPLOAD PATENT DOCUMENT**", 
        type = ['csv', 'xlsx', 'xls'],
        help = "파일 형식 : CSV, EXCEL",
        key = "data_upload"
    )

    df = None

    if uploaded_file is not None:

        try:

            file_extension = uploaded_file.name.split('.')[-1].lower()
            
            if file_extension == 'csv':
                df = pd.read_csv(uploaded_file)

            elif file_extension in ['xlsx', 'xls']:

                excel_file = pd.ExcelFile(uploaded_file)

                sheet_names = excel_file.sheet_names
                
                if len(sheet_names) > 1:
                    selected_sheet = st.sidebar.selectbox("**CHOOSE SHEET**", sheet_names)
                else:
                    selected_sheet = sheet_names[0]
                
                df = pd.read_excel(uploaded_file, sheet_name = selected_sheet)
            
            st.session_state.uploaded_df = df
            
            with st.expander("**DATA PREVIEW**", expanded = True):

                st.dataframe(df.head(), width = "stretch")
                
                st.metric("**TOTAL ROWS**", len(df))
            
                with st.expander("**COLUMN NAMES**", expanded = False):
                    st.write(list(df.columns))
            
        except Exception as e:
            st.sidebar.info(f"🔴 {e}")

    if st.session_state.uploaded_df is not None:
        st.sidebar.markdown("---")
        sidebar_title("APPROACH")
        classification_method = st.sidebar.selectbox(
            "**SELECT CLASSIFICATION METHOD**",
            ["-- SELECT --", "PROMPT ENGINEERING", "FINE TUNING"]
        )

    else:
        classification_method = None
        st.info("⬅️ 특허 문서를 업로드해 주세요.")
        
    return classification_method