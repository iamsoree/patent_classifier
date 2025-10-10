# tab/history.py

import streamlit as st
from util.database.ft_database import TrainingDatabase
import plotly.express as px
import os
import shutil
from io import BytesIO
import pandas as pd
import time

def show():
    
    st.markdown(
        "<h3 style = 'text-decoration: underline; text-decoration-thickness: 3px; text-underline-offset: 7px; text-decoration-color: slategray; margin-bottom: 20px;'>OVERALL SUMMARY</h3>",
        unsafe_allow_html = True
    )
    
    db = TrainingDatabase()
    
    records_df = db.get_records_dataframe()

    records_df = records_df.reset_index(drop = True)
    records_df['display_idx'] = records_df.index + 1 

    with st.container():
        col1, col2, col3 = st.columns(3)
        for col, title, value, color in zip(
            [col1, col2, col3],
            ["TRAINING RUNS", "MODELS USED", "DATA PROCESSED"],
            [len(records_df), records_df['model_name'].nunique(), f"{records_df['data_count'].sum():,}"],
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
                        margin-bottom: 45px;
                    ">
                        <div style = "font-size: 17px; color: {color}; font-weight: 750;">{title}</div>
                        <div style = "font-size: 25px; font-weight: 700; color: dimgray;">{value}</div>
                    </div>
                    """,
                    unsafe_allow_html = True
                )

    if not records_df.empty and 'accuracy' in records_df.columns:

        st.markdown(
            "<h3 style = 'text-decoration: underline; text-decoration-thickness: 3px; text-underline-offset: 7px; text-decoration-color: slategray;'>ACCURACY BY MODEL</h3>",
            unsafe_allow_html = True
        )

        fig = px.bar(
            records_df, 
            x = 'display_idx', 
            y = 'accuracy',
            color = 'model_name',
            hover_data = ['id', 'timestamp', 'data_count'],
            labels = {'idx' : 'INDEX', 'accuracy' : 'ACCURACY', 'model_name' : 'MODEL NAME'},
            color_discrete_sequence = px.colors.sequential.Blues[::-1]
        )
        
        fig.update_layout(
            xaxis_title = "INDEX",
            yaxis_title = "ACCURACY",
            yaxis = dict(tickformat = '.1%'),
            height = 450,
            showlegend = True,
            legend = dict(
                orientation = "v",
                yanchor = "top",
                y = 1,
                xanchor = "left",
                x = 1.03
            )
        )
        
        st.plotly_chart(fig, width = 'stretch')
    
    st.markdown("---")

    if not records_df.empty:

        with st.expander("**PREVIOUS RECORD**", expanded = True):
        
            col_type, col_filter, col_sort = st.columns([1, 3, 1])

            with col_type:
                model_type_filter = st.selectbox(
                    "**FILTER BY PARADIGM**",
                    options = ['ALL', 'REPRESENTATION', 'GENERATIVE'],
                    index = 0,
                    key = "history_model_type_filter"
                )

            if model_type_filter == 'ALL':
                type_filtered_df = records_df.copy()
            else:
                type_filtered_df = records_df[records_df['model_type'] == model_type_filter].copy()
            
            with col_filter:
                model_options = list(type_filtered_df['model_name'].unique())
                selected_models = st.multiselect(
                    "**FILTER BY MODEL**",
                    options = model_options,
                    default = model_options
                )
            
            with col_sort:
                sort_options = {
                    'TIME ASCENDING': ('timestamp', True),
                    'TIME DESCENDING': ('timestamp', False),
                    'ACCURACY ASCENDING': ('accuracy', True),
                    'ACCURACY DESCENDING': ('accuracy', False)
                }            
                selected_sort = st.selectbox(
                    "**SORT BY**",
                    options = list(sort_options.keys()),
                    index = 0
                )
            
            filtered_df = type_filtered_df.copy()

            if selected_models:
                filtered_df = filtered_df[filtered_df['model_name'].isin(selected_models)]
            else:
                filtered_df = filtered_df.iloc[0:0]
            
            sort_column, ascending = sort_options[selected_sort]
            filtered_df = filtered_df.sort_values(by = sort_column, ascending = ascending).reset_index(drop = True)
            filtered_df['display_idx'] = filtered_df.index + 1

            display_df = filtered_df if not records_df.empty else records_df

            for idx, record in display_df.iterrows():

                col_expander, col_delete = st.columns([17.5, 1])
                
                with col_expander:
                    with st.expander(
                        f"**[{record['display_idx']}] {record['timestamp']}&nbsp;&nbsp;|&nbsp;&nbsp;{record['model_name']}&nbsp;&nbsp;|&nbsp;&nbsp;{record.get('model_type', 'GENERATIVE')}&nbsp;&nbsp;|&nbsp;&nbsp;{record['data_count']} DATA ITEMS&nbsp;&nbsp;|&nbsp;&nbsp;(ACCURACY) {record['accuracy'] * 100:.1f}%**",
                        expanded = False
                    ):
                        
                        detailed_record = db.get_record_by_id(record['id'])
                        
                        if detailed_record:

                            with st.expander("**BASIC INFORMATION**", expanded = False):
                                col1, col2 = st.columns(2)                       
                                with col1:
                                    st.write(f"**[ DATE ]**<br>{detailed_record['timestamp']}", unsafe_allow_html = True)
                                    st.write(f"**[ MODEL NAME ]**<br>{detailed_record['model_name']}", unsafe_allow_html = True)
                                    st.write(f"**[ MODEL TYPE ]**<br>{detailed_record.get('model_type', 'GENERATIVE')}", unsafe_allow_html = True)                    
                                with col2:
                                    st.write(f"**[ DATA COUNT ]**<br>{detailed_record['data_count']:,}", unsafe_allow_html = True)
                                    st.write(f"**[ MODEL STORAGE LOCATION ]**<br>{detailed_record['output_path']}", unsafe_allow_html = True)
                            
                            with st.expander("**LABEL INFORMATION**", expanded = False):
                                col1, col2 = st.columns(2)
                                labels = detailed_record['label_types']
                                with col1:
                                    st.write(f"**[ LABEL COUNT ]**<br>{len(labels)}", unsafe_allow_html = True)
                                with col2:
                                    st.write(f"**[ LABEL LIST ]**<br>{', '.join(labels)}", unsafe_allow_html = True)
                            
                            with st.expander("**COULUMN INFORMATION**", expanded = False):
                                col1, col2 = st.columns(2)
                                columns = detailed_record['text_columns']
                                with col1:
                                    st.write(f"**[ COULUMN COUNT ]**<br>{len(columns)}", unsafe_allow_html = True)
                                with col2:
                                    st.write(f"**[ COULUMN LIST ]**<br>{', '.join(columns)}", unsafe_allow_html = True)
                            
                            with st.expander("**HYPERPARAMETER**", expanded = False):
                                hyperparams = detailed_record['hyperparameters']
                                model_type = hyperparams.get('model_type', 'GENERATIVE')
                                if model_type == 'GENERATIVE':  
                                    col1, col2, col3 = st.columns(3)                    
                                    with col1:
                                        st.write(f"**[ EPOCHS ]**<br>{hyperparams.get('num_train_epochs', 'N/A')}", unsafe_allow_html = True)
                                        st.write(f"**[ LEARNING RATE ]**<br>{hyperparams.get('learning_rate', 'N/A')}", unsafe_allow_html = True)
                                        st.write(f"**[ TRAIN BATCH SIZE ]**<br>{hyperparams.get('per_device_train_batch_size', 'N/A')}", unsafe_allow_html = True)
                                    
                                    with col2:
                                        st.write(f"**[ MAX LENGTH ]**<br>{hyperparams.get('max_length', 'N/A')}", unsafe_allow_html = True)
                                        st.write(f"**[ LoRA RANK (R) ]**<br>{hyperparams.get('lora_r', 'N/A')}", unsafe_allow_html = True)
                                        st.write(f"**[ LoRA ALPHA ]**<br>{hyperparams.get('lora_alpha', 'N/A')}", unsafe_allow_html = True)
                                    
                                    with col3:
                                        st.write(f"**[ WARMUP STEPS ]**<br>{hyperparams.get('warmup_steps', 'N/A')}", unsafe_allow_html = True)
                                        st.write(f"**[ GRADIENT ACCUMULATION STEPS ]**<br>{hyperparams.get('gradient_accumulation_steps', 'N/A')}", unsafe_allow_html = True)
                                        st.write(f"**[ OPTIM ]**<br>{hyperparams.get('optim', 'N/A')}", unsafe_allow_html = True)
                                else:
                                    col1, col2, col3 = st.columns(3)
                                    with col1:
                                        st.write(f"**[ EPOCHS ]**<br>{hyperparams.get('num_train_epochs', 'N/A')}", unsafe_allow_html = True)
                                        st.write(f"**[ LEARNING RATE ]**<br>{hyperparams.get('learning_rate', 'N/A')}", unsafe_allow_html = True)
                                        st.write(f"**[ TRAIN BATCH SIZE ]**<br>{hyperparams.get('per_device_train_batch_size', 'N/A')}", unsafe_allow_html = True)
                                    
                                    with col2:
                                        st.write(f"**[ MAX LENGTH ]**<br>{hyperparams.get('max_length', 'N/A')}", unsafe_allow_html = True)
                                        st.write(f"**[ WEIGHT DECAY ]**<br>{hyperparams.get('weight_decay', 'N/A')}", unsafe_allow_html = True)
                                    
                                    with col3:
                                        st.write(f"**[ WARMUP STEPS ]**<br>{hyperparams.get('warmup_steps', 'N/A')}", unsafe_allow_html = True)
                                        st.write(f"**[ GRADIENT ACCUMULATION STEPS ]**<br>{hyperparams.get('gradient_accumulation_steps', 'N/A')}", unsafe_allow_html = True)
                            
                            with st.expander("**CLASSIFICATION REPORT**", expanded = False):
                                results = detailed_record['results']                        
                                col1, col2, col3, col4 = st.columns(4)                      
                                with col1:
                                    accuracy = results.get('eval_accuracy', 0)
                                    st.metric("**ACCURACY**", f"{accuracy:.2f}")                       
                                with col2:
                                    f1_score = results.get('eval_f1', 0)
                                    st.metric("**F1 SCORE**", f"{f1_score:.2f}")                        
                                with col3:
                                    precision = results.get('eval_precision', 0)
                                    st.metric("**PRECISION**", f"{precision:.2f}")                       
                                with col4:
                                    recall = results.get('eval_recall', 0)
                                    st.metric("**RECALL**", f"{recall:.2f}")
                
                with col_delete:

                    if st.button(f"✖️", key = f"delete_{record['id']}", type = "secondary", help = "삭제"):

                        model_path = record['output_path']
                        if os.path.exists(model_path):
                            if os.path.isdir(model_path):
                                shutil.rmtree(model_path)

                        if db.delete_record(record['id']):
                            st.rerun()
    
    st.markdown("---")

    excel_buffer = BytesIO()

    with pd.ExcelWriter(excel_buffer, engine = 'openpyxl') as writer:
        
        display_df.to_excel(writer, index = False, sheet_name = "History")

    st.download_button(
        label = "✔️ DOWNLOAD HISTORY (EXCEL)",
        data = excel_buffer.getvalue(),
        file_name = f"history_{int(time.time())}.xlsx",
        mime = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )