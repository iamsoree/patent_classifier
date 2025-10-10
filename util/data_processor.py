# util/data_processor.py

import pandas as pd
from datasets import Dataset, DatasetDict

class DataProcessor:

    @staticmethod
    def validate_dataframe(df):

        if df is None or len(df) == 0:
            raise ValueError("NO DATA AVAILABLE.")

        return True

    @staticmethod
    def get_available_columns(df, exclude_cols = None):

        if exclude_cols is None:
            exclude_cols = ["사용자태그", "WINTELIPS KEY"]

        return [col for col in df.columns if col not in exclude_cols]
    
    @staticmethod
    def prepare_data(executor, df, selected_cols = None, label_col = None):

        if selected_cols is None:
            selected_cols = ["발명의 명칭", "요약", "전체청구항"]

        def combine_text(row):
            text_parts = []
            for col in selected_cols:
                if col in row.index and pd.notna(row.get(col, "")):
                    text_parts.append(f"{col} : {row[col]}")
            return " ".join(text_parts)

        df_copy = df.copy()

        df_copy["combined_text"] = df_copy.apply(combine_text, axis = 1)

        if label_col and label_col in df_copy.columns:

            executor.labels_list = sorted(df_copy[label_col].unique())
            executor.label2id = {l : i for i, l in enumerate(executor.labels_list)}
            executor.id2label = {i : l for l, i in executor.label2id.items()}

            processed_df = pd.DataFrame({
                "text" : df_copy["combined_text"],
                "labels" : df_copy[label_col],
                "patent_id" : df_copy["출원번호"]
            })

            processed_df["label"] = processed_df["labels"].map(executor.label2id)

        else:
            processed_df = pd.DataFrame({
                "text" : df_copy["combined_text"],
                "patent_id" : df_copy["출원번호"]
            })

        return processed_df
    
    @staticmethod
    def create_balanced_datasetdict(df_chunked, tokenizer, test_size = 0.2, random_state = 25):

        label_counts = df_chunked['label'].value_counts()

        min_count = label_counts.min()

        train_samples_per_label = int(min_count * (1 - test_size))
        test_samples_per_label = min_count - train_samples_per_label

        print(f"(FOR EACH LABEL) TRAIN : {train_samples_per_label}개, TEST : {test_samples_per_label}개")

        train_dfs = []
        test_dfs = []

        for label in sorted(df_chunked['label'].unique()):

            label_data = df_chunked[df_chunked['label'] == label].sample(
                n = min_count,
                random_state = random_state
            ).reset_index(drop = True)

            train_data = label_data.iloc[:train_samples_per_label]
            test_data = label_data.iloc[train_samples_per_label:]

            train_dfs.append(train_data)
            test_dfs.append(test_data)

        train_df = pd.concat(train_dfs, ignore_index = True).sample(frac = 1, random_state = random_state)
        test_df = pd.concat(test_dfs, ignore_index = True).sample(frac = 1, random_state = random_state)

        train_dataset_dict = {
            'text' : train_df['text'].tolist(),
            'label' : train_df['label'].tolist()
        }
        test_dataset_dict = {
            'text' : test_df['text'].tolist(),
            'label' : test_df['label'].tolist()
        }

        train_data = Dataset.from_dict(train_dataset_dict)
        test_data = Dataset.from_dict(test_dataset_dict)

        dataset = DatasetDict({
            'train' : train_data,
            'test' : test_data
        })

        def preprocess_function(examples):
            tokenized = tokenizer(examples['text'], truncation = True, max_length = 512)
            tokenized['labels'] = [int(l) for l in examples['label']]
            return tokenized

        tokenized_dataset = dataset.map(preprocess_function, batched = True)
        tokenized_dataset = tokenized_dataset.remove_columns(['text', 'label'])

        return tokenized_dataset, test_df