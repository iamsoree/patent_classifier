# util/finetuning_train.py

from dotenv import load_dotenv
import os
from transformers import AutoTokenizer, AutoModelForSequenceClassification, BitsAndBytesConfig, DataCollatorWithPadding
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import torch
import pandas as pd
import torch.nn.functional as F
from datasets import DatasetDict
from transformers import TrainingArguments, Trainer
from trl import SFTConfig, SFTTrainer
import pickle

load_dotenv()

class FineTuningTrain:

    def __init__(self, model_name = "google/gemma-2-2b", hf_token = None, model_type = None):
        self.model_name = model_name
        self.hf_token = hf_token or os.getenv('HF_TOKEN')
        self.tokenizer = None
        self.model = None
        self.trainer = None
        self.labels_list = None
        self.label2id = None
        self.id2label = None
        self.test_df = None
        self.model_type = model_type

    def initialize_tokenizer(self):
        
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            token = self.hf_token,
            trust_remote_code = True
        )

        self.tokenizer.padding_side = 'right'

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def setup_model(self, bnb_config_params = None, lora_config_params = None):

        if self.model_type == 'REPRESENTATION':
            self.model = AutoModelForSequenceClassification.from_pretrained(
                self.model_name,
                token = self.hf_token,
                num_labels = len(self.labels_list),
                id2label = self.id2label,
                label2id = self.label2id,
                trust_remote_code = True
            )
            self.model.config.pad_token_id = self.tokenizer.pad_token_id
            return None
        
        else:

            if bnb_config_params is None:
                bnb_config_params = {
                    'load_in_4bit' : True,
                    'bnb_4bit_quant_type' : 'nf4',
                    'bnb_4bit_compute_dtype' : 'float16',
                    'bnb_4bit_use_double_quant' : True
                }

            bnb_config = BitsAndBytesConfig(**bnb_config_params)

            self.model = AutoModelForSequenceClassification.from_pretrained(
                self.model_name,
                token = self.hf_token,
                num_labels = len(self.labels_list),
                device_map = 'auto',
                quantization_config = bnb_config
            )

            self.model.config.use_cache = False
            self.model.config.pad_token_id = self.tokenizer.pad_token_id

            if lora_config_params is None:
                lora_config_params = {
                    'lora_alpha' : 128,
                    'lora_dropout' : 0.1,
                    'r' : 64,
                    'bias' : 'none',
                    'task_type' : 'SEQ_CLS',
                    'target_modules' : ['k_proj', 'gate_proj', 'v_proj', 'up_proj', 'q_proj', 'o_proj', 'down_proj']
                }

            peft_config = LoraConfig(**lora_config_params)

            self.model = prepare_model_for_kbit_training(self.model)
            self.model = get_peft_model(self.model, peft_config)

            return peft_config

    def compute_metrics(self, pred):

        # labels = pred.label_ids
        # preds = pred.predictions.argmax(-1)

        # precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average = 'macro')
        # acc = accuracy_score(labels, preds)

        # logits_tensor = torch.tensor(pred.predictions)
        # labels_tensor = torch.tensor(pred.label_ids)

        # loss = F.cross_entropy(logits_tensor, labels_tensor).item()

        # return {
        #     'accuracy' : acc,
        #     'f1' : f1,
        #     'precision' : precision,
        #     'recall' : recall,
        #     'eval_loss' : loss
        # }

        logits = torch.tensor(pred.predictions)
        probs = torch.softmax(logits, dim = -1).numpy()

        test_df_copy = self.test_df.reset_index(drop = True)
        test_df_copy['chunk_index'] = range(len(test_df_copy))

        patent_results = []

        for patent_id, group in test_df_copy.groupby('patent_id'):

            indices = group['chunk_index'].tolist()
            
            group = group.copy()
            predictions_for_chunks = probs[indices]
            mean_prob = predictions_for_chunks.mean(axis = 0)
            
            pred_idx = mean_prob.argmax()
            pred_label = self.id2label[pred_idx]
            
            patent_results.append({
                "patent_id" : patent_id,
                "pred_label" : pred_label,
                "true_label" : self.id2label[group['label'].iloc[0]]
            })

        patent_results_df = pd.DataFrame(patent_results)
        
        true_labels = patent_results_df["true_label"].tolist()
        pred_labels = patent_results_df["pred_label"].tolist()

        precision, recall, f1, _ = precision_recall_fscore_support(true_labels, pred_labels, average = "weighted")
        acc = accuracy_score(true_labels, pred_labels)

        labels_tensor = torch.tensor(pred.label_ids)
        loss = F.cross_entropy(logits, labels_tensor).item()

        return {
            'accuracy' : acc,
            'f1' : f1,
            'precision' : precision,
            'recall' : recall,
            'eval_loss' : loss
        }

    def train_model(self, tokenized_dataset, test_df, output_dir, bnb_config_params = None, lora_config_params = None, training_config_params = None, use_balanced_split = True):

        self.test_df = test_df

        if isinstance(tokenized_dataset, DatasetDict):
            tokenized_train = tokenized_dataset['train']
            tokenized_test = tokenized_dataset['test']

        else:
            tokenized_train, tokenized_test = tokenized_dataset

        data_collator = DataCollatorWithPadding(tokenizer = self.tokenizer)

        default_training_args = {
            'output_dir' : output_dir,
            'learning_rate' : 2e-5 if self.model_type == 'GENERATIVE' else 5e-5,
            'per_device_train_batch_size' : 2 if self.model_type == 'GENERATIVE' else 4,
            'per_device_eval_batch_size' : 2 if self.model_type == 'GENERATIVE' else 4,
            'gradient_accumulation_steps' : 2 if self.model_type == 'GENERATIVE' else 4,
            'num_train_epochs' : 5,
            'warmup_steps' : 50,
            'logging_steps' : 10,
            'fp16' : True,
            'gradient_checkpointing' : True,
            'label_names' : ['labels']
        }

        if self.model_type == 'REPRESENTATION':
            default_training_args['weight_decay'] = 0.01
        else:
            default_training_args['max_length'] = 512
            default_training_args['dataset_text_field'] = 'text'
            default_training_args['optim'] = 'paged_adamw_32bit'
            default_training_args['lr_scheduler_type'] = 'cosine'

        if training_config_params:
            filtered_params = {k : v for k, v in training_config_params.items() if v is not None}
            default_training_args.update(filtered_params)

        if self.model_type == 'REPRESENTATION':

            training_arguments = TrainingArguments(**default_training_args)

            self.trainer = Trainer(
                model = self.model,
                args = training_arguments,
                train_dataset = tokenized_train,
                eval_dataset = tokenized_test,
                processing_class = self.tokenizer,
                data_collator = data_collator,
                compute_metrics = self.compute_metrics
            )

        else:

            training_arguments = SFTConfig(**default_training_args)

            if lora_config_params is None:
                lora_config_params = {
                    'lora_alpha' : 128,
                    'lora_dropout' : 0.1,
                    'r' : 64,
                    'bias' : 'none',
                    'task_type' : 'SEQ_CLS',
                    'target_modules' : ['k_proj', 'gate_proj', 'v_proj', 'up_proj', 'q_proj', 'o_proj', 'down_proj']
                }

            peft_config = LoraConfig(**lora_config_params)

            self.trainer = SFTTrainer(
                model = self.model,
                train_dataset = tokenized_train,
                eval_dataset = tokenized_test,
                processing_class = self.tokenizer,
                args = training_arguments,
                data_collator = data_collator,
                compute_metrics = self.compute_metrics,
                peft_config = peft_config
            )

        self.trainer.train()

        os.makedirs(output_dir, exist_ok = True)
        with open(os.path.join(output_dir, 'label_mappings.pkl'), 'wb') as f:
            pickle.dump({
                'labels_list' : self.labels_list,
                'label2id' : self.label2id,
                'id2label' : self.id2label,
                'model_type' : self.model_type
            }, f)

        return self.trainer.evaluate()

    def save_model(self, output_dir, merge_adapter = True):

        if self.trainer:

            if self.model_type == 'REPRESENTATION':
                self.trainer.model.save_pretrained(output_dir)
                self.tokenizer.save_pretrained(output_dir)
                print(output_dir)

            else:

                if merge_adapter:
                    merged_model = self.trainer.model.merge_and_unload()
                    merged_output_dir = os.path.join(output_dir, "merge_model")
                    os.makedirs(merged_output_dir, exist_ok = True)
                    merged_model.save_pretrained(merged_output_dir)              
                    self.tokenizer.save_pretrained(merged_output_dir)
                    print(merged_output_dir)

                else:
                    self.trainer.model.save_pretrained(output_dir)
                    self.tokenizer.save_pretrained(output_dir)