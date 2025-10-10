# util/database/ft_database.py

import sqlite3
from datetime import datetime
import uuid
import json
import pandas as pd

class TrainingDatabase:
    
    def __init__(self, db_path = "ft_history.db"):
        self.db_path = db_path
        self.init_database()
    
    def init_database(self):

        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS ft_history (
                idx INTEGER PRIMARY KEY AUTOINCREMENT,
                id TEXT UNIQUE NOT NULL,
                timestamp TEXT NOT NULL,
                model_name TEXT NOT NULL,
                model_type TEXT DEFAULT 'GENERATIVE',
                data_count INTEGER NOT NULL,
                label_types TEXT NOT NULL,
                text_columns TEXT NOT NULL,
                hyperparameters TEXT NOT NULL,
                output_path TEXT NOT NULL,
                results TEXT NOT NULL,
                label_mappings TEXT NOT NULL,
                max_length INTEGER DEFAULT 512,
                stride INTEGER DEFAULT 50
            )
        ''')

        cursor.execute("PRAGMA table_info(ft_history)")
        columns = [col[1] for col in cursor.fetchall()]
        
        if 'model_type' not in columns:
            cursor.execute('''
                ALTER TABLE ft_history 
                ADD COLUMN model_type TEXT DEFAULT 'GENERATIVE'
            ''')
        
        conn.commit()
        conn.close()
    
    def save_training_record(self, model_name, data_count, labels_list, text_columns, hyperparameters, output_path, results, label_mappings):

        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        id = str(uuid.uuid4())

        max_length = hyperparameters.get('max_length', 512)
        stride = hyperparameters.get('stride', 50)
        model_type = hyperparameters.get('model_type', 'GENERATIVE')
        
        cursor.execute('''
            INSERT INTO ft_history 
            (id, timestamp, model_name, model_type, data_count, label_types, text_columns, hyperparameters, output_path, results, label_mappings, max_length, stride)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            id,
            timestamp,
            model_name,
            model_type,
            data_count,
            json.dumps(labels_list, ensure_ascii = False),
            json.dumps(text_columns, ensure_ascii = False),
            json.dumps(hyperparameters, ensure_ascii = False),
            output_path,
            json.dumps(results, ensure_ascii = False),
            json.dumps(label_mappings, ensure_ascii = False),
            max_length,
            stride
        ))
        
        conn.commit()
        conn.close()
        
        return id
    
    def get_all_records(self):

        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("PRAGMA table_info(ft_history)")
        columns = [col[1] for col in cursor.fetchall()]
        has_model_type = 'model_type' in columns
        
        if has_model_type:
            cursor.execute('''
                SELECT idx, id, timestamp, model_name, model_type, data_count, label_types, text_columns, output_path, results
                FROM ft_history
                ORDER BY timestamp ASC
            ''')
        else:
            cursor.execute('''
                SELECT idx, id, timestamp, model_name, data_count, label_types, text_columns, output_path, results
                FROM ft_history
                ORDER BY timestamp ASC
            ''')
        
        records = cursor.fetchall()
        conn.close()
        
        return records
    
    def get_record_by_id(self, id):

        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("PRAGMA table_info(ft_history)")
        columns = [col[1] for col in cursor.fetchall()]
        has_model_type = 'model_type' in columns
        
        cursor.execute('SELECT * FROM ft_history WHERE id = ?', (id,))
        
        record = cursor.fetchone()
        conn.close()
        
        if record:
            if has_model_type:
                return {
                    'idx' : record[0],
                    'id' : record[1],
                    'timestamp' : record[2],
                    'model_name' : record[3],
                    'data_count' : record[4],
                    'label_types' : json.loads(record[5]),
                    'text_columns' : json.loads(record[6]),
                    'hyperparameters' : json.loads(record[7]),
                    'output_path' : record[8],
                    'results' : json.loads(record[9]),
                    'label_mappings' : json.loads(record[10]),
                    'max_length' : record[11] if len(record) > 11 else 512,
                    'stride' : record[12] if len(record) > 12 else 50,
                    'model_type' : record[13] if len(record) > 13 else 'GENERATIVE'
                }
            else:
                return {
                    'idx' : record[0],
                    'id' : record[1],
                    'timestamp' : record[2],
                    'model_name' : record[3],
                    'model_type' : 'GENERATIVE',
                    'data_count' : record[4],
                    'label_types' : json.loads(record[5]),
                    'text_columns' : json.loads(record[6]),
                    'hyperparameters' : json.loads(record[7]),
                    'output_path' : record[8],
                    'results' : json.loads(record[9]),
                    'label_mappings' : json.loads(record[10]),
                    'max_length' : record[11] if len(record) > 11 else 512,
                    'stride' : record[12] if len(record) > 12 else 50
                }
        
        return None
    
    def get_record_by_path(self, model_path):

        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("PRAGMA table_info(ft_history)")
        columns = [col[1] for col in cursor.fetchall()]
        has_model_type = 'model_type' in columns

        if model_path.endswith('merge_model'):
            parent_path = model_path.replace('\\merge_model', '').replace('/merge_model', '')
            cursor.execute('SELECT * FROM ft_history WHERE output_path = ?', (parent_path,))
        else:
            cursor.execute('SELECT * FROM ft_history WHERE output_path = ?', (model_path,))
        
        record = cursor.fetchone()
        conn.close()
        
        if record:
            if has_model_type:
                return {
                    'idx' : record[0],
                    'id' : record[1],
                    'timestamp' : record[2],
                    'model_name' : record[3],
                    'data_count' : record[4],
                    'label_types' : json.loads(record[5]),
                    'text_columns' : json.loads(record[6]),
                    'hyperparameters' : json.loads(record[7]),
                    'output_path' : record[8],
                    'results' : json.loads(record[9]),
                    'label_mappings' : json.loads(record[10]),
                    'max_length' : record[11] if len(record) > 11 else 512,
                    'stride' : record[12] if len(record) > 12 else 50,
                    'model_type' : record[13] if len(record) > 13 else 'GENERATIVE'
                }
            else:
                return {
                    'idx' : record[0],
                    'id' : record[1],
                    'timestamp' : record[2],
                    'model_name' : record[3],
                    'model_type' : 'GENERATIVE',
                    'data_count' : record[4],
                    'label_types' : json.loads(record[5]),
                    'text_columns' : json.loads(record[6]),
                    'hyperparameters' : json.loads(record[7]),
                    'output_path' : record[8],
                    'results' : json.loads(record[9]),
                    'label_mappings' : json.loads(record[10]),
                    'max_length' : record[11] if len(record) > 11 else 512,
                    'stride' : record[12] if len(record) > 12 else 50
                }
        
        return None
    
    def delete_record(self, id):

        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('DELETE FROM ft_history WHERE id = ?', (id,))
        affected_rows = cursor.rowcount
        
        conn.commit()
        conn.close()
        
        return affected_rows > 0
    
    def get_records_dataframe(self):

        conn = sqlite3.connect(self.db_path)

        cursor = conn.cursor()
        cursor.execute("PRAGMA table_info(ft_history)")
        columns = [col[1] for col in cursor.fetchall()]
        has_model_type = 'model_type' in columns

        if has_model_type:
            query = '''
                SELECT idx, id, timestamp, model_name, model_type, data_count, label_types, text_columns, output_path, results
                FROM ft_history
                ORDER BY timestamp ASC
            '''
        else:
            query = '''
                SELECT idx, id, timestamp, model_name, data_count, label_types, text_columns, output_path, results
                FROM ft_history
                ORDER BY timestamp ASC
            '''
        
        df = pd.read_sql_query(query, conn)
        
        conn.close()
        
        if not df.empty:

            if not has_model_type:
                df['model_type'] = 'GENERATIVE'

            df['label_count'] = df['label_types'].apply(lambda x : len(json.loads(x)))
            df['column_count'] = df['text_columns'].apply(lambda x : len(json.loads(x)))
            
            def extract_accuracy(results_json):
                try:
                    results = json.loads(results_json)
                    return results.get('eval_accuracy', 0)
                except:
                    return 0
            
            df['accuracy'] = df['results'].apply(extract_accuracy)
        
        return df