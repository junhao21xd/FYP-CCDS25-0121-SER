import json
import pandas as pd
from sklearn.model_selection import StratifiedKFold
import os

'''
Preprocesses the dataset into 5 isolated sets for training AudEERING VAD models to generate Out-of-Fold (OOF) predictions for downstream LLM training.
Subset 5 maintains the exact data distribution of the original dataset, but preprocessed into the exact same format for seamless VAD model evaluation.
'''

def standardise_and_save(df_sub_train, df_sub_test, output_train_csv_path, output_test_csv_path, target_cols, FIELD_MAP, max_vad):
    train_vals = (df_sub_train[target_cols].values - 1) / max_vad
    test_vals  = (df_sub_test[target_cols].values - 1) / max_vad
    
    train_vals = train_vals.clip(0, 1)
    test_vals = test_vals.clip(0, 1)
    
    df_sub_train['labels'] = train_vals.tolist()
    df_sub_test['labels'] = test_vals.tolist()
    
    # 5. Save new JSON with only necessary fields
    output_train_data = df_sub_train[[FIELD_MAP['id'], FIELD_MAP['path'], 'labels']].to_dict(orient='records')
    output_test_data = df_sub_test[[FIELD_MAP['id'], FIELD_MAP['path'], 'labels']].to_dict(orient='records')
    
    with open(output_train_csv_path, 'w') as f:
        json.dump(output_train_data, f, indent=2)
        
    with open(output_test_csv_path, 'w') as f:
        json.dump(output_test_data, f, indent=2)
        
def preprocess_iemocap_audeering(input_path, output_data_path, generate_oof=True):
    print(f"Processing {input_path}...")
    os.makedirs(output_data_path, exist_ok=True)
    FIELD_MAP = {
        'id': 'id',
        'valence': 'valence', 
        'arousal': 'arousal', 
        'dominance': 'dominance',
        'path': 'path' 
    }
    max_vad = 4.0
    
    # 1. Load Data
    try:
        df = pd.read_csv(input_path)
    except ValueError:
        # Fallback if json is line-separated (JSONL)
        df = pd.read_json(input_path, lines=True)

    train_df = df[df['split']=='train'].copy()
    test_df = df[df['split']=='test'].copy()
    
    v_col = FIELD_MAP['valence']
    a_col = FIELD_MAP['arousal']
    d_col = FIELD_MAP['dominance']
    
    target_cols = [a_col, d_col, v_col]

    if generate_oof:
        for ses in range(1,5):
            df_sub_train = train_df[train_df['session']!=ses].copy()
            df_sub_test = train_df[train_df['session']==ses].copy()
            
            output_train_csv_path = f'{output_data_path}/train_vad_ready_sess{ses}.json'
            output_test_csv_path = f'{output_data_path}/test_vad_ready_sess{ses}.json'

            standardise_and_save(df_sub_train, df_sub_test, output_train_csv_path, output_test_csv_path, target_cols, FIELD_MAP, max_vad)
    
    output_train_csv_path = f'{output_data_path}/train_vad_ready_sess5.json'
    output_test_csv_path = f'{output_data_path}/test_vad_ready_sess5.json'
    
    standardise_and_save(train_df, test_df, output_train_csv_path, output_test_csv_path, target_cols, FIELD_MAP, max_vad)
        
def preprocess_msp_audeering(input_path, output_data_path, generate_oof=True):
    print(f"Processing {input_path}...")
    # FIELD_MAP = {
    #     'id': 'file',
    #     'valence': 'EmoVal', 
    #     'arousal': 'EmoAct', 
    #     'dominance': 'EmoDom',
    #     'path': 'audio_filepath',
    #     'emotion': 'major_emotion'
    # }
    FIELD_MAP = {
        'id': 'id',
        'valence': 'valence', 
        'arousal': 'arousal', 
        'dominance': 'dominance',
        'path': 'path',
        'emotion': 'emotion'
    }
    max_vad = 6.0
    
    # Extract relevant columns
    v_col = FIELD_MAP['valence']
    a_col = FIELD_MAP['arousal']
    d_col = FIELD_MAP['dominance']
        
    target_cols = [a_col, d_col, v_col]
    
    # 1. Load Data
    try:
        df = pd.read_csv(input_path)
    except ValueError:
        # Fallback if json is line-separated (JSONL)
        df = pd.read_json(input_path, lines=True)

    df_train = df[df['split']=='train'].copy()
    df_ses5 = df[df['split']=='test'].copy()

    if generate_oof:
        y = df_train[FIELD_MAP['emotion']]
        skf = StratifiedKFold(n_splits=4, shuffle=True, random_state=42)
        for i, (train_index, val_index) in enumerate(skf.split(df_train, y)): 
            X_train_fold = df_train.iloc[train_index].copy()
            X_val_fold = df_train.iloc[val_index].copy()
        
            output_train_csv_path = f'./{output_data_path}/train_vad_ready_sess{i+1}.json'
            output_test_csv_path = f'./{output_data_path}/test_vad_ready_sess{i+1}.json'

            standardise_and_save(X_train_fold, X_val_fold, output_train_csv_path, output_test_csv_path, target_cols, FIELD_MAP, max_vad)
    
    output_train_csv_path = f'./{output_data_path}/train_vad_ready_sess5.json'
    output_test_csv_path = f'./{output_data_path}/test_vad_ready_sess5.json'

    standardise_and_save(df_train, df_ses5, output_train_csv_path, output_test_csv_path, target_cols, FIELD_MAP, max_vad)