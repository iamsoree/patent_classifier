# util/finetuning_inference.py

from dotenv import load_dotenv
import os
import pickle
from util.model_detector import detect_model_type
from transformers import AutoTokenizer, AutoModelForSequenceClassification, BitsAndBytesConfig, DataCollatorWithPadding
from peft import PeftModel
import torch
from util.data_processor import DataProcessor
from util.text_chunker import SlidingWindowChunker
import traceback
from datasets import Dataset
from torch.utils.data import DataLoader
import pandas as pd

load_dotenv()

class FineTuningInference:

    def __init__(self, model_name = "google/gemma-2-2b", hf_token = None, model_type = None):
        self.model_name = model_name
        self.hf_token = hf_token or os.getenv('HF_TOKEN')
        self.tokenizer = None
        self.model = None
        self.labels_list = None
        self.label2id = None
        self.id2label = None
        self.model_type = model_type or detect_model_type(model_name)
        
    def load_model(self, model_path, manual_labels = None, is_merged_model = False):

        if not model_path or not os.path.exists(model_path):
            raise ValueError(model_path)

        merged_model_path = os.path.join(model_path, "merge_model")

        if not is_merged_model and os.path.exists(merged_model_path):
            model_path = merged_model_path
            is_merged_model = True

        label_file = os.path.join(model_path, 'label_mappings.pkl')

        if os.path.exists(label_file):
            try:
                with open(label_file, 'rb') as f:
                    mappings = pickle.load(f)
                    self.labels_list = mappings['labels_list']
                    self.label2id = mappings['label2id']
                    self.id2label = mappings['id2label']
            except Exception as e:
                raise ValueError(e)
            
        elif manual_labels:
            self.labels_list = sorted(manual_labels)
            self.label2id = {l : i for i, l in enumerate(self.labels_list)}
            self.id2label = {i : l for l, i in self.label2id.items()}

        else:
            self.labels_list = ['CPC_C01B', 'CPC_C01C', 'CPC_C01D', 'CPC_C01F', 'CPC_C01G']
            self.label2id = {l : i for i, l in enumerate(self.labels_list)}
            self.id2label = {i : l for l, i in self.label2id.items()}

        try:
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_path,
                token = self.hf_token,
                trust_remote_code = True
            )

        except Exception:
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                token = self.hf_token,
                trust_remote_code = True
            )

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.tokenizer.padding_side = 'right'

        try:

            if self.model_type == 'REPRESENTATION':
                self.model = AutoModelForSequenceClassification.from_pretrained(
                    model_path,
                    token = self.hf_token,
                    num_labels = len(self.labels_list),
                    trust_remote_code = True
                )
                print("(REPRESENTATION) MODEL LOADED.")

            else:

                if is_merged_model:

                    self.model = AutoModelForSequenceClassification.from_pretrained(
                        model_path,
                        token = self.hf_token,
                        num_labels = len(self.labels_list),
                        device_map = 'auto',
                        trust_remote_code = True,
                        torch_dtype = torch.float16
                    )

                    print("(GENERATIVE) MERGED MODEL LOADED.")

                else:

                    bnb_config = BitsAndBytesConfig(
                        load_in_4bit = True,
                        bnb_4bit_quant_type = 'nf4',
                        bnb_4bit_compute_dtype = 'float16',
                        bnb_4bit_use_double_quant = True
                    )

                    base_model = AutoModelForSequenceClassification.from_pretrained(
                        self.model_name,
                        token = self.hf_token,
                        num_labels = len(self.labels_list),
                        device_map = 'auto',
                        quantization_config = bnb_config,
                        trust_remote_code = True
                    )

                    self.model = PeftModel.from_pretrained(
                        base_model,
                        model_path,
                        device_map = 'auto'
                    )

                    self.model = self.model.merge_and_unload()

                    print("(GENERATIVE) LOADED AFTER MERGING.")

            self.model.eval()

        except Exception as e:
            raise ValueError(e)

    def predict_patents(self, df, model_path = None, selected_cols = None, max_length = 512, stride = 50):

        try:

            if model_path and not self.model:
                self.load_model(model_path)

            if not self.model or not self.tokenizer:
                raise ValueError("THE MODEL HAS NOT BEEN LOADED.")

            if selected_cols is None:
                selected_cols = ["발명의 명칭", "요약", "전체청구항"]

            processed_df = DataProcessor.prepare_data(self, df, selected_cols, label_col = None)
            
            chunker = SlidingWindowChunker(self.tokenizer)
            
            df_chunked = chunker.create_chunked_dataset(processed_df, max_length, stride)

        except Exception as e:
            print(e)
            traceback.print_exc()
            raise

        try:

            test_data = Dataset.from_pandas(df_chunked)

            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token

            self.tokenizer.padding_side = 'right'

            if getattr(self.model.config, 'pad_token_id', None) is None:
                self.model.config.pad_token_id = self.tokenizer.pad_token_id

            def preprocess_function(examples):
                tokenized = self.tokenizer(
                    examples['text'],
                    truncation = True,
                    max_length = max_length,
                    padding = True
                )
                return tokenized

            tokenized_test = test_data.map(preprocess_function, batched = True)

            remove_cols = ['text', 'patent_id']

            if 'label' in df_chunked.columns:
                remove_cols.append('label')

            tokenized_test = tokenized_test.remove_columns(remove_cols)

            data_collator = DataCollatorWithPadding(tokenizer = self.tokenizer, padding = True)

            dataloader = DataLoader(tokenized_test, batch_size = 2, collate_fn = data_collator)

            self.model.eval()

            all_predictions = []

            with torch.no_grad():

                for batch in dataloader:

                    if next(self.model.parameters()).device.type == 'cuda':
                        batch = {k : v.cuda() if isinstance(v, torch.Tensor) else v for k, v in batch.items()}

                    outputs = self.model(**batch)

                    logits = outputs.logits

                    predictions = torch.softmax(logits, dim = -1)

                    all_predictions.append(predictions.cpu())

            probs = torch.cat(all_predictions, dim = 0).numpy()

        except Exception as e:
            print(e)
            traceback.print_exc()
            raise

        try:

            df_chunked = df_chunked.reset_index(drop = True)

            df_chunked['chunk_index'] = range(len(df_chunked))

            patent_results = []

            for patent_id, group in df_chunked.groupby('patent_id'):

                indices = group['chunk_index'].tolist()

                group = group.copy()

                predictions_for_chunks = probs[indices]

                mean_prob = predictions_for_chunks.mean(axis = 0)

                pred_idx = mean_prob.argmax()

                pred_label = self.id2label[pred_idx]

                patent_results.append({
                    "출원_번호" : patent_id,
                    "분류_코드" : pred_label,
                    "신뢰도" : round(mean_prob[pred_idx], 4)
                })

            return pd.DataFrame(patent_results)

        except Exception as e:
            print(e)
            traceback.print_exc()
            raise