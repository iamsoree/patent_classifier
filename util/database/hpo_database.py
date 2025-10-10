# util/database/hpo_database.py

import sqlite3
from datetime import datetime
import uuid
import json
import pandas as pd

class HPODatabase:
    
    def __init__(self, db_path = "hpo_history.db"):
        self.db_path = db_path
        self.init_database()
    
    def init_database(self):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS hpo_history (
                idx INTEGER PRIMARY KEY AUTOINCREMENT,
                id TEXT UNIQUE NOT NULL,
                timestamp TEXT NOT NULL,
                model_name TEXT NOT NULL,
                model_type TEXT DEFAULT 'GENERATIVE',
                data_count INTEGER NOT NULL,
                label_types TEXT NOT NULL,
                text_columns TEXT NOT NULL,
                n_trials INTEGER NOT NULL,
                best_hyperparams TEXT NOT NULL,
                best_results TEXT NOT NULL,
                all_trials TEXT NOT NULL,
                study_name TEXT NOT NULL
            )
        ''')

        cursor.execute("PRAGMA table_info(hpo_history)")
        columns = [col[1] for col in cursor.fetchall()]
        
        if 'model_type' not in columns:
            cursor.execute('''
                ALTER TABLE hpo_history 
                ADD COLUMN model_type TEXT DEFAULT 'GENERATIVE'
            ''')
        
        conn.commit()
        conn.close()
    
    def save_hpo_record(self, model_name, data_count, labels_list, text_columns, n_trials, best_hyperparams, best_results, all_trials, study_name, model_type = 'GENERATIVE'):
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        id = str(uuid.uuid4())
        
        cursor.execute('''
            INSERT INTO hpo_history 
            (id, timestamp, model_name, model_type, data_count, label_types, text_columns, n_trials, best_hyperparams, best_results, all_trials, study_name)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            id,
            timestamp,
            model_name,
            model_type,
            data_count,
            json.dumps(labels_list, ensure_ascii = False),
            json.dumps(text_columns, ensure_ascii = False),
            n_trials,
            json.dumps(best_hyperparams, ensure_ascii = False),
            json.dumps(best_results, ensure_ascii = False),
            json.dumps(all_trials, ensure_ascii = False),
            study_name
        ))
        
        conn.commit()
        conn.close()
        
        return id
    
    def cleanup_old_records(self):
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        try:

            cursor.execute('''
                WITH ranked_records AS (
                    SELECT id,
                           model_name,
                           JSON_EXTRACT(best_results, '$.accuracy') as accuracy,
                           ROW_NUMBER() OVER (
                               PARTITION BY model_name 
                               ORDER BY JSON_EXTRACT(best_results, '$.accuracy') DESC
                           ) as rn
                    FROM hpo_history
                )
                SELECT id FROM ranked_records WHERE rn = 1
            ''')
            
            best_record_ids = [row[0] for row in cursor.fetchall()]
            
            if not best_record_ids:
                conn.close()
                return 0
            
            placeholders = ','.join(['?' for _ in best_record_ids])
            cursor.execute(f'''
                DELETE FROM hpo_history 
                WHERE id NOT IN ({placeholders})
            ''', best_record_ids)
            
            deleted_count = cursor.rowcount
            
            conn.commit()
            print(f"{deleted_count} RECORDS DELETED")
            
            return deleted_count
            
        except Exception as e:
            print(str(e))
            conn.rollback()
            return 0
            
        finally:
            conn.close()
    
    def get_best_record_by_model(self):

        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("PRAGMA table_info(hpo_history)")
        columns = [col[1] for col in cursor.fetchall()]
        has_model_type = 'model_type' in columns
        
        cursor.execute('''
            WITH ranked_records AS (
                SELECT *,
                       ROW_NUMBER() OVER (
                           PARTITION BY model_name 
                           ORDER BY JSON_EXTRACT(best_results, '$.accuracy') DESC
                       ) as rn
                FROM hpo_history
            )
            SELECT * FROM ranked_records WHERE rn = 1
            ORDER BY model_name
        ''')
        
        records = cursor.fetchall()
        conn.close()
        
        best_records = []

        for record in records:
            try:
                if has_model_type:
                    best_results = json.loads(record[9]) if isinstance(record[9], str) else record[9]
                    best_hyperparams = json.loads(record[8]) if isinstance(record[8], str) else record[8]
                    best_records.append({
                        'idx' : record[0],
                        'id' : record[1],
                        'timestamp' : record[2],
                        'model_name' : record[3],
                        'model_type' : record[12] if len(record) > 12 else 'GENERATIVE',
                        'data_count' : record[4],
                        'label_types' : json.loads(record[5]) if isinstance(record[5], str) else record[5],
                        'text_columns' : json.loads(record[6]) if isinstance(record[6], str) else record[6],
                        'n_trials' : record[7],
                        'best_hyperparams' : best_hyperparams,
                        'best_results' : best_results,
                        'study_name' : record[11]
                    })
                else:
                    best_results = json.loads(record[9]) if isinstance(record[9], str) else record[9]
                    best_hyperparams = json.loads(record[8]) if isinstance(record[8], str) else record[8]
                    best_records.append({
                        'idx' : record[0],
                        'id' : record[1],
                        'timestamp' : record[2],
                        'model_name' : record[3],
                        'model_type' : 'GENERATIVE',
                        'data_count' : record[4],
                        'label_types' : json.loads(record[5]) if isinstance(record[5], str) else record[5],
                        'text_columns' : json.loads(record[6]) if isinstance(record[6], str) else record[6],
                        'n_trials' : record[7],
                        'best_hyperparams' : best_hyperparams,
                        'best_results' : best_results,
                        'study_name' : record[11]
                    })
            except json.JSONDecodeError:
                continue
                
        return best_records
    
    def get_all_records(self):

        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("PRAGMA table_info(hpo_history)")
        columns = [col[1] for col in cursor.fetchall()]
        has_model_type = 'model_type' in columns
        
        if has_model_type:
            cursor.execute('''
                SELECT idx, id, timestamp, model_name, model_type, data_count, n_trials, best_results, study_name
                FROM hpo_history
                ORDER BY timestamp DESC
            ''')
        else:
            cursor.execute('''
                SELECT idx, id, timestamp, model_name, data_count, n_trials, best_results, study_name
                FROM hpo_history
                ORDER BY timestamp DESC
            ''')
        
        records = cursor.fetchall()
        conn.close()
        
        return records
    
    def get_record_by_id(self, id):

        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("PRAGMA table_info(hpo_history)")
        columns = [col[1] for col in cursor.fetchall()]
        has_model_type = 'model_type' in columns
        
        cursor.execute('SELECT * FROM hpo_history WHERE id = ?', (id,))
        
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
                    'n_trials' : record[7],
                    'best_hyperparams' : json.loads(record[8]),
                    'best_results' : json.loads(record[9]),
                    'all_trials' : json.loads(record[10]),
                    'study_name' : record[11],
                    'model_type' : record[12] if len(record) > 12 else 'GENERATIVE'
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
                    'n_trials' : record[7],
                    'best_hyperparams' : json.loads(record[8]),
                    'best_results' : json.loads(record[9]),
                    'all_trials' : json.loads(record[10]),
                    'study_name' : record[11]
                }
        
        return None
    
    def delete_record(self, id):

        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('DELETE FROM hpo_history WHERE id = ?', (id,))
        affected_rows = cursor.rowcount
        
        conn.commit()
        conn.close()
        
        return affected_rows > 0
    
    def get_records_dataframe(self):

        conn = sqlite3.connect(self.db_path)

        cursor = conn.cursor()
        cursor.execute("PRAGMA table_info(hpo_history)")
        columns = [col[1] for col in cursor.fetchall()]
        has_model_type = 'model_type' in columns

        if has_model_type:
            query = '''
                SELECT idx, id, timestamp, model_name, model_type, data_count, n_trials, best_results, study_name
                FROM hpo_history
                ORDER BY timestamp DESC
            '''
        else:
            query = '''
                SELECT idx, id, timestamp, model_name, data_count, n_trials, best_results, study_name
                FROM hpo_history
                ORDER BY timestamp DESC
            '''
        
        df = pd.read_sql_query(query, conn)
        
        conn.close()
        
        if not df.empty:

            if not has_model_type:
                df['model_type'] = 'GENERATIVE'

            def extract_accuracy(results_json):
                try:
                    results = json.loads(results_json)
                    return results.get('accuracy', 0)
                except:
                    return 0
            
            df['accuracy'] = df['best_results'].apply(extract_accuracy)
        
        return df