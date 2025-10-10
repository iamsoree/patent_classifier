# util/model_detector.py

import os
import json

def detect_model_type(model_name_or_type):
    
    representation_keywords = [
        'bert', 'roberta', 'distilbert', 'albert', 
        'electra', 'deberta', 'scibert', 'biobert',
        'camembert', 'xlm-roberta', 'cased'
    ]
    
    generative_keywords = [
        'gpt', 'gemma', 'llama', 'qwen', 'phi', 
        'mistral', 'falcon', 'mpt', 'opt', 'bloom'
    ]
    
    text_lower = str(model_name_or_type).lower()
    
    for keyword in representation_keywords:
        if keyword in text_lower:
            return "REPRESENTATION"
    
    for keyword in generative_keywords:
        if keyword in text_lower:
            return "GENERATIVE"
    
    return "GENERATIVE"

def is_representation_model(model_path):

    adapter_config = os.path.join(model_path, 'adapter_config.json')
    if os.path.exists(adapter_config):
        return False
    
    config_path = os.path.join(model_path, 'config.json')
    if os.path.exists(config_path):
        try:
            with open(config_path, 'r') as f:
                config = json.load(f)
                model_type = config.get('model_type', '')
                return detect_model_type(model_type) == "REPRESENTATION"
        except:
            pass
    
    return False