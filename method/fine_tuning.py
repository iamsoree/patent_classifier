# method/fine_tuning.py

from dotenv import load_dotenv
import streamlit as st
import os
from tab import inference, train, history, hpo

load_dotenv()

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
                FINE TUNING
            </h2>
            ''',
            unsafe_allow_html = True
        )

        tab1, tab2, tab3, tab4 = st.tabs(["**INFERENCE**", "**TRAIN**", "**HYPERPARAMETER OPTIMIZATION**", "**HISTORY**"])

        with tab1:
            inference.show()

        with tab2:
            train.show()

        with tab3:
            hpo.show()

        with tab4:
            history.show()

    else:
        st.info("⬅️ 특허 문서를 업로드해 주세요.")