# util/optuna_tuner.py

from dotenv import load_dotenv
import os
from util.database.ft_database import TrainingDatabase
from transformers import AutoTokenizer, AutoModelForSequenceClassification, BitsAndBytesConfig, DataCollatorWithPadding
from datetime import datetime
import optuna
import torch
import gc
from transformers import TrainingArguments, Trainer
from trl import SFTConfig, SFTTrainer
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import pandas as pd
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
import json
import optuna.visualization as vis

load_dotenv()

class OptunaHyperparameterTuner:
    
    def __init__(self, model_name = "google/gemma-2-2b", hf_token = None, model_type = None):
        self.model_name = model_name
        self.hf_token = hf_token or os.getenv('HF_TOKEN')
        self.tokenizer = None
        self.labels_list = None
        self.label2id = None
        self.id2label = None
        self.tokenized_dataset = None
        self.test_df = None
        self.base_output_dir = None
        self.study = None
        self.db = TrainingDatabase()
        self.model_type = model_type
        
    def initialize_tokenizer(self):
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            token = self.hf_token,
            trust_remote_code = True
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = 'right'
        
    def set_data(self, tokenized_dataset, test_df, labels_list, label2id, id2label):
        self.tokenized_dataset = tokenized_dataset
        self.test_df = test_df
        self.labels_list = labels_list
        self.label2id = label2id
        self.id2label = id2label
        
    def create_study(self, study_name = None, direction = 'maximize', storage = None):

        if study_name is None:
            study_name = f"hpo_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            
        self.study = optuna.create_study(
            direction = direction,
            study_name = study_name,
            storage = storage,
            load_if_exists = True
        )
        
        return self.study
        
    def objective(self, trial, base_output_dir, search_space_config):

        try:

            torch.cuda.empty_cache()
            gc.collect()

            hyperparams = self._suggest_hyperparams(trial, search_space_config)
            
            trial_output_dir = os.path.join(base_output_dir, f"trial_{trial.number}")
            os.makedirs(trial_output_dir, exist_ok = True)
            
            model = self._create_model(hyperparams)
            
            training_args = self._create_training_args(hyperparams, trial_output_dir)
            
            data_collator = DataCollatorWithPadding(tokenizer = self.tokenizer)

            def compute_metrics(pred):
                return self._compute_metrics(pred)
            
            if self.model_type == 'REPRESENTATION':
                trainer = Trainer(
                    model = model,
                    args = training_args,
                    train_dataset = self.tokenized_dataset['train'],
                    eval_dataset = self.tokenized_dataset['test'],
                    processing_class = self.tokenizer,
                    data_collator = data_collator,
                    compute_metrics = compute_metrics
                )

            else:
                trainer = SFTTrainer(
                    model = model,
                    args = training_args,
                    train_dataset = self.tokenized_dataset['train'],
                    eval_dataset = self.tokenized_dataset['test'],
                    processing_class = self.tokenizer,
                    data_collator = data_collator,
                    compute_metrics = compute_metrics,
                    peft_config = None
                )

            trainer.train()

            preds_output = trainer.predict(self.tokenized_dataset['test'])
            logits = torch.tensor(preds_output.predictions)
            probs = torch.softmax(logits, dim = -1).numpy()

            eval_results = self._compute_patent_level_metrics(probs)

            trial.set_user_attr('accuracy', eval_results['eval_accuracy'])
            trial.set_user_attr('f1', eval_results['eval_f1'])
            trial.set_user_attr('precision', eval_results['eval_precision'])
            trial.set_user_attr('recall', eval_results['eval_recall'])
            
            self._save_trial_results(trial, hyperparams, eval_results, trial_output_dir)
            
            target_metric = eval_results.get('eval_accuracy', 0.0)
            
            print(f"[TRIAL_{trial.number}] ACCURACY : {target_metric:.4f}")

            del trainer
            del model
            torch.cuda.empty_cache()
            gc.collect()
            
            return target_metric
            
        except Exception as e:
            print(f"[TRIAL_{trial.number}] ERROR : {str(e)}")
            torch.cuda.empty_cache()
            gc.collect()
            return None
        
    def _compute_patent_level_metrics(self, probs):

        try:

            test_df = self.test_df.reset_index(drop = True)
            test_df['chunk_index'] = range(len(test_df))

            patent_results = []

            for patent_id, group in test_df.groupby('patent_id'):
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

            results_df = pd.DataFrame(patent_results)
            
            true_labels = results_df["true_label"].tolist()
            pred_labels = results_df["pred_label"].tolist()
            
            accuracy = accuracy_score(true_labels, pred_labels)
            precision, recall, f1, _ = precision_recall_fscore_support(true_labels, pred_labels, average = "macro", zero_division = 0)
            
            return {
                'eval_accuracy' : accuracy,
                'eval_f1' : f1,
                'eval_precision' : precision,
                'eval_recall' : recall
            }
            
        except Exception as e:
            print(e)
            return {
                'eval_accuracy' : 0.0,
                'eval_f1' : 0.0,
                'eval_precision' : 0.0,
                'eval_recall' : 0.0
            }
            
    def _suggest_hyperparams(self, trial, search_space_config):

        hyperparams = {}
        
        if 'learning_rate' in search_space_config:
            config = search_space_config['learning_rate']
            hyperparams['learning_rate'] = trial.suggest_float(
                'learning_rate', 
                config.get('low', 1e-7), 
                config.get('high', 5e-3), 
                log = config.get('log', True)
            )
        
        if 'batch_size' in search_space_config:
            config = search_space_config['batch_size']
            hyperparams['per_device_train_batch_size'] = trial.suggest_categorical(
                'batch_size', config.get('choices', [1, 2, 4, 8, 16])
            )
            hyperparams['per_device_eval_batch_size'] = hyperparams['per_device_train_batch_size']

        if 'epochs' in search_space_config:
            config = search_space_config['epochs']
            hyperparams['num_train_epochs'] = trial.suggest_int(
                'epochs', 
                config.get('low', 1), 
                config.get('high', 25)
            )
        
        if 'lora_r' in search_space_config:
            config = search_space_config['lora_r']
            hyperparams['lora_r'] = trial.suggest_categorical(
                'lora_r', config.get('choices', [8, 16, 32, 64, 128])
            )
        
        if 'lora_alpha' in search_space_config:
            config = search_space_config['lora_alpha']
            hyperparams['lora_alpha'] = trial.suggest_categorical(
                'lora_alpha', config.get('choices', [16, 32, 64, 128, 256])
            )
        
        if 'lora_dropout' in search_space_config:
            config = search_space_config['lora_dropout']
            hyperparams['lora_dropout'] = trial.suggest_float(
                'lora_dropout', 
                config.get('low', 0.0), 
                config.get('high', 0.5)
            )
        
        if 'warmup_steps' in search_space_config:
            config = search_space_config['warmup_steps']
            hyperparams['warmup_steps'] = trial.suggest_int(
                'warmup_steps', 
                config.get('low', 0), 
                config.get('high', 100)
            )

        if 'gradient_accumulation_steps' in search_space_config:
            config = search_space_config['gradient_accumulation_steps']
            hyperparams['gradient_accumulation_steps'] = trial.suggest_categorical(
                'gradient_accumulation_steps', config.get('choices', [1, 2, 4, 8, 16, 32])
            )

        if 'weight_decay' in search_space_config:
            config = search_space_config['weight_decay']
            hyperparams['weight_decay'] = trial.suggest_float(
                'weight_decay', 
                config.get('low', 0.0), 
                config.get('high', 0.1)
            )
        
        return hyperparams
    
    def _create_model(self, hyperparams):

        if self.model_type == 'REPRESENTATION':

            model = AutoModelForSequenceClassification.from_pretrained(
                self.model_name,
                token = self.hf_token,
                num_labels = len(self.labels_list),
                trust_remote_code = True
            )
            
            model.config.use_cache = False

            if self.tokenizer.pad_token_id is not None:
                model.config.pad_token_id = self.tokenizer.pad_token_id
        
            return model
        
        else:

            bnb_config = BitsAndBytesConfig(
                load_in_4bit = True,
                bnb_4bit_quant_type = 'nf4',
                bnb_4bit_compute_dtype = 'float16',
                bnb_4bit_use_double_quant = True
            )
            
            model = AutoModelForSequenceClassification.from_pretrained(
                self.model_name,
                token = self.hf_token,
                num_labels = len(self.labels_list),
                device_map = 'auto',
                quantization_config = bnb_config,
                trust_remote_code = True
            )
            
            model.config.use_cache = False

            if self.tokenizer.pad_token_id is not None:
                model.config.pad_token_id = self.tokenizer.pad_token_id

            model = prepare_model_for_kbit_training(model)

            peft_config = LoraConfig(
                lora_alpha = hyperparams.get('lora_alpha', 128),
                lora_dropout = hyperparams.get('lora_dropout', 0.1),
                r = hyperparams.get('lora_r', 64),
                bias = 'none',
                task_type = 'SEQ_CLS',
                target_modules = ['k_proj', 'gate_proj', 'v_proj', 'up_proj', 'q_proj', 'o_proj', 'down_proj']
            )
            
            model = get_peft_model(model, peft_config)
            
            return model
    
    def _create_training_args(self, hyperparams, output_dir):

        base_args = {
            'output_dir' : output_dir,
            'learning_rate' : hyperparams.get('learning_rate', 2e-5 if self.model_type == 'GENERATIVE' else 5e-5),
            'per_device_train_batch_size' : hyperparams.get('per_device_train_batch_size', 2 if self.model_type == 'GENERATIVE' else 4),
            'per_device_eval_batch_size' : hyperparams.get('per_device_eval_batch_size', 2 if self.model_type == 'GENERATIVE' else 4),
            'gradient_accumulation_steps' : hyperparams.get('gradient_accumulation_steps', 2 if self.model_type == 'GENERATIVE' else 4),
            'num_train_epochs' : hyperparams.get('num_train_epochs', 3),
            'warmup_steps' : hyperparams.get('warmup_steps', 50),
            'logging_steps' : 10,
            'save_steps' : 500,
            'fp16' : True,
            'gradient_checkpointing' : True,
            'label_names' : ['labels'],
            'save_total_limit' : 1,
            'dataloader_pin_memory' : False
        }

        if self.model_type == 'REPRESENTATION':
            base_args['weight_decay'] = hyperparams.get('weight_decay', 0.01)
            return TrainingArguments(**base_args)
        
        else:
            base_args['max_length'] = 512
            base_args['dataset_text_field'] = 'text'
            base_args['optim'] = 'paged_adamw_32bit'
            base_args['lr_scheduler_type'] = 'cosine'
            return SFTConfig(**base_args)
    
    def _compute_metrics(self, pred):

        try:

            if hasattr(pred, 'predictions') and pred.predictions is not None:
                logits = pred.predictions
            else:
                return None
            
            if hasattr(pred, 'label_ids') and pred.label_ids is not None:
                labels = pred.label_ids
            else:
                return None

            predictions = logits.argmax(axis = -1)

            acc = accuracy_score(labels, predictions)
            precision, recall, f1, _ = precision_recall_fscore_support(labels, predictions, average = "macro", zero_division = 0)
            
            return {
                'accuracy' : float(acc),
                'f1' : float(f1),
                'precision' : float(precision),
                'recall' : float(recall)
            }
            
        except Exception as e:
            print(e)
            return None
    
    def _save_trial_results(self, trial, hyperparams, eval_results, output_dir):

        trial_info = {
            'trial_number' : trial.number,
            'hyperparams' : hyperparams,
            'results' : eval_results,
            'timestamp' : datetime.now().isoformat()
        }
        
        results_file = os.path.join(output_dir, 'trial_results.json')

        with open(results_file, 'w', encoding = 'utf-8') as f:
            json.dump(trial_info, f, ensure_ascii = False, indent = 2)
    
    def optimize(self, base_output_dir, search_space_config, n_trials = 25):

        self.base_output_dir = base_output_dir

        if self.study is None:
            self.create_study()
        
        def objective_wrapper(trial):
            return self.objective(trial, base_output_dir, search_space_config)
        
        self.study.optimize(objective_wrapper, n_trials = n_trials)
        
        return self.study
    
    def get_best_params(self):

        if self.study is None:
            return None
        
        return self.study.best_params
    
    def get_best_value(self):

        if self.study is None:
            return None
        
        return self.study.best_value
    
    def get_best_trial_metrics(self):

        if self.study is None:
            return None
            
        best_trial = self.study.best_trial

        if best_trial and hasattr(best_trial, 'user_attrs'):
            return {
                'accuracy' : best_trial.user_attrs.get('accuracy', best_trial.value),
                'f1' : best_trial.user_attrs.get('f1', 0.0),
                'precision' : best_trial.user_attrs.get('precision', 0.0),
                'recall' : best_trial.user_attrs.get('recall', 0.0)
            }
        
        return None
    
    def create_optimization_plots(self):

        if self.study is None:
            return []
        
        plots = []
        
        try:

            fig1 = vis.plot_optimization_history(self.study)
            plots.append(fig1)

            fig2 = vis.plot_param_importances(self.study)
            plots.append(fig2)
            
        except Exception as e:
            print(e)
        
        return plots
    
    def save_study_results(self, output_path):

        if self.study is None:
            return
        
        results = {
            'best_params' : self.study.best_params,
            'best_value' : self.study.best_value,
            'n_trials' : len(self.study.trials),
            'study_name' : self.study.study_name,
            'trials' : []
        }

        for trial in self.study.trials:
            trial_info = {
                'number' : trial.number,
                'value' : trial.value,
                'params' : trial.params,
                'state' : trial.state.name,
                'datetime_start' : trial.datetime_start.isoformat() if trial.datetime_start else None,
                'datetime_complete' : trial.datetime_complete.isoformat() if trial.datetime_complete else None,
                'user_attrs' : trial.user_attrs if hasattr(trial, 'user_attrs') else {}
            }
            results['trials'].append(trial_info)
        
        with open(output_path, 'w', encoding = 'utf-8') as f:
            json.dump(results, f, ensure_ascii = False, indent = 2)