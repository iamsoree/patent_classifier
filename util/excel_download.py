# util/excel_download.py

from io import BytesIO
import pandas as pd
import streamlit as st
import time

def show_unified(results_df, classification_groups):
    
    excel_buffer = BytesIO()

    with pd.ExcelWriter(excel_buffer, engine = 'openpyxl') as writer:

        results_df.to_excel(writer, sheet_name = '전체', index = False)
 
        for category, group in classification_groups:
            safe_name = category.replace('/', '_').replace(':', '_')[:31]
            group.to_excel(writer, sheet_name = safe_name, index = False)

        stats_df = results_df['분류 코드'].value_counts().reset_index()
        stats_df.columns = ['분류 코드', '개수']
        stats_df.to_excel(writer, sheet_name = '통계', index = False)

    excel_buffer.seek(0)

    st.download_button(
        label = "✔️ DOWNLOAD CLASSIFICATION RESULT (EXCEL)",
        data = excel_buffer.getvalue(),
        file_name = f"patent_classification_{int(time.time())}.xlsx",
        mime = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )

def show_promptengineering(results_df, classification_groups):
    show_unified(results_df, classification_groups)

def show_finetuning(results_df, classification_groups):
    show_unified(results_df, classification_groups)