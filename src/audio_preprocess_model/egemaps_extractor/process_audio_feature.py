import pandas as pd
import os
import numpy as np
from configs.dataset_config import FEATURE_RENAMING_MAPS, DATASET_PROMPT_GROUPS, EXTRACTED_FEATURE_SET

def extract_thresholds_and_stats(df, dataset, num_classes):
    
    features = EXTRACTED_FEATURE_SET[dataset]
    thresholds = {}
    stats = {}
    
    # Calculate overall thresholds and stats
    overall_thresholds = {}
    overall_stats = {}
    for feature in features:
        feature_data = df[feature].replace(0, np.nan).dropna()
        if num_classes == 5:
            q1, q2, q3, q4 = feature_data.quantile([0.1, 0.25, 0.75, 0.9])
            overall_thresholds[feature] = {'very_low': q1, 'low': q2, 'medium': q3, 'high': q4}
        overall_stats[feature] = {'mean': feature_data.mean(), 'std': feature_data.std()}
    
    thresholds['overall'] = overall_thresholds
    stats['overall'] = overall_stats
    
    # Calculate gender-specific thresholds, stats, and plot
    gender_list = df['gender'].unique().tolist()
    for gender in gender_list:
        gender_df = df[df['gender'] == gender]
        gender_thresholds = {}
        gender_stats = {}
        for feature in features:
            feature_data = gender_df[feature].replace(0, np.nan).dropna()
            if len(feature_data) > 0:
                if num_classes == 5:
                    q1, q2, q3, q4 = feature_data.quantile([0.1, 0.25, 0.75, 0.9])
                    gender_thresholds[feature] = {'very_low': q1, 'low': q2, 'medium': q3, 'high': q4}
                gender_stats[feature] = {'mean': feature_data.mean(), 'std': feature_data.std()}
            else:
                gender_thresholds[feature] = overall_thresholds[feature]
                gender_stats[feature] = overall_stats[feature]
        thresholds[gender] = gender_thresholds
        stats[gender] = gender_stats
    
    return thresholds, stats

def categorize(value, thresholds, num_classes):
    if pd.isna(value) or value == 0:
        return 'none'

    elif num_classes == 5:
        if value <= thresholds['very_low']:
            return 'Very low'
        elif value <= thresholds['low']:
            return 'Low'
        elif value <= thresholds['medium']:
            return 'Medium'
        elif value <= thresholds['high']:
            return 'High'
        else:
            return 'Very high'

def standardize_and_process_df(df, thresholds, stats, num_classes):
    features = list(thresholds['overall'].keys())
    
    # Standardize features
    for feature in features:
        df[f'{feature}_standardized'] = df.apply(lambda row: 
            (row[feature] - stats.get(row['gender'], stats['overall'])[feature]['mean']) / 
            stats.get(row['gender'], stats['overall'])[feature]['std']
        if not pd.isna(row[feature]) and row[feature] != 0 else np.nan, axis=1)
    
    # Categorize original features
    for feature in features:
        df[f'{feature}_category'] = df.apply(lambda row: categorize(
            row[feature], 
            thresholds.get(row['gender'], thresholds['overall'])[feature],
            num_classes
        ), axis=1)
    
    return df


def add_conversation_history(df, window_size=8):
    """
    Creates a 'history_str' for IEMOCAP using strict logic from data_process.py.
    
    Logic:
      - Window: Current utterance + previous 'window_size' utterances.
      - Format: Tab-separated (\t), matching the original script.
      - Speakers: Uses df['gender'] (e.g., 'M'/'F'). 
        (Note: Original script mapped M->0, F->1. If you prefer that, 
         you can map the column before running this.)
    """
    # 1. Sort to ensure temporal order
    # Assuming 'segment_id' or 'turn_index' exists to order utterances within a dialogue
    df = df.sort_values(by=['video_id', 'Order_Index']) 
    
    # Initialize column
    df['history_context'] = ""
    
    # 2. Iterate through each distinct conversation
    for conversation_id, group in df.groupby('video_id', sort=False):
        texts = group['text'].tolist()
        # Use the explicit gender column as requested
        speakers = group['gender'].tolist()
        indices = group.index.tolist()
        pitches = group['Average Pitch_category'].tolist()
        variations = group['Pitch Stability (StdDev)_category'].tolist()
 
        for i, row_idx in enumerate(indices):
            start_pos = max(0, i - window_size)
            end_pos = i + 1
            
            # Slice the lists
            w_texts = texts[start_pos:end_pos]
            w_speakers = speakers[start_pos:end_pos]
            w_pitches = pitches[start_pos:end_pos]
            w_variations = variations[start_pos:end_pos]

            lines = []
            window_len = len(w_texts)

            # Iterate through the current window slice
            for k in range(window_len):
                s = w_speakers[k]
                u = w_texts[k]
                p = w_pitches[k]
                v = w_variations[k]

                # Base string
                utterance_str = f'Speaker_{s}:"{u}"'

                # Add features only to the last 3 items
                # "k" is the index within this specific window (0 to window_len-1).
                # If window has 5 items, indices are 0,1,2,3,4. We want 2,3,4.
                if k >= window_len - 3:
                    utterance_str += f' ({p} pitch with {v} variation)'
                
                lines.append(utterance_str)

            df.at[row_idx, 'history_context'] = "\t " + "\t ".join(lines)

    return df

def add_one_line_convo(processed_df):
    # temp_content_str = 'The following noted between \'### ###\' is a single isolated utterance with its speech features attached. ### '
    # temp_content_str += f"\t Speaker_{row['gender']}: {row['transcription']}"
    # temp_content_str += f" ({row['Average Pitch_category']} pitch with {row['Pitch Stability (StdDev)_category']} variation)  ### \n"

    # return temp_content_str
    prefix = "The following noted between \'### ###\' is a single isolated utterance with its speech features attached. ### "
    
    # Vectorized string concatenation
    processed_df['history_context'] = (
        prefix + 
        "\t Speaker_" + processed_df['gender'].astype(str) + ": " + 
        processed_df['text'] + 
        " (" + processed_df['Average Pitch_category'] + " pitch with " + 
        processed_df['Pitch Stability (StdDev)_category'] + " variation)  ### \n"
    )
    
def prepare_and_save_json(df, dataset, output_path):
    print(f"Processing {len(df)} rows...")
    final_columns = {} # Dict to map {Old_Name : New_Name}
    
    group_order = DATASET_PROMPT_GROUPS.get(dataset)
    
    # 2a. Add Metadata columns
    final_columns['id'] = 'id'
    final_columns['text'] = 'utterance'
    final_columns['emotion'] = 'output'
    final_columns['path'] = 'path'
    final_columns['history_context'] = 'history_context'

    if 'pred_valence' in df.columns: 
        df.drop(columns=['valence'], inplace=True)
        final_columns['pred_valence'] = 'valence'
        
    if 'pred_arousal' in df.columns: 
        df.drop(columns=['arousal'], inplace=True)
        final_columns['pred_arousal'] = 'arousal'
        
    if 'pred_dominance' in df.columns:
        df.drop(columns=['dominance'], inplace=True)
        final_columns['pred_dominance'] = 'dominance'
    
    # 2b. Add Acoustic columns (and ensure they exist)
    for group, features in group_order.items():
        for feature in features:
            # Check if your DF has "Average Pitch" OR "Average Pitch_category"
            if f"{feature}_category" in df.columns:
                final_columns[f"{feature}_category"] = f"{feature}_category"
            elif feature in df.columns:
                final_columns[feature] = f"{feature}_category" # Rename it!
            else:
                # If missing, create a placeholder so code doesn't crash
                print(f"Warning: Missing {feature}, filling with 'N/A'")
                df[f"{feature}_category"] = "N/A"
                final_columns[f"{feature}_category"] = f"{feature}_category"   
    
    iemocap_to_target = {
        'hap': 'happy',
        'happy': 'happy',

        'sad': 'sad',

        'neu': 'neutral',
        'neutral': 'neutral',

        'ang': 'angry',
        'anger': 'angry',

        'exc': 'excited',
        'excited': 'excited',

        'fru': 'frustrated',
        'frustrated': 'frustrated',
    }
    df['emotion'] = df['emotion'].map(iemocap_to_target) 
    
    export_df = df[list(final_columns.keys())].rename(columns=final_columns)
    
    export_df.to_json(
        output_path, 
        orient='records', 
        indent=4, 
        force_ascii=False
    )
    
    print(f"Saved to {output_path}")

def process_audio_feature(dataset, input_csv, output_path):    
    output_train_json = os.path.join(output_path, 'train.json')
    output_test_json = os.path.join(output_path, 'test.json')
    
    # Load the dataset
    df = pd.read_csv(input_csv)
    df.rename(columns=FEATURE_RENAMING_MAPS[dataset], inplace=True)
    num_classes = 5  
    
    # Extract thresholds and stats based on the training data
    train_df = df[df['split'] == 'train'].copy()
    thresholds, stats = extract_thresholds_and_stats(train_df, dataset, num_classes)
    
    # Process the entire dataset
    processed_df = standardize_and_process_df(df, thresholds, stats, num_classes)
    
    if dataset == "msp":
        processed_df = add_one_line_convo(processed_df)

    elif dataset == "iemocap":
        processed_df = add_conversation_history(processed_df)

    # Create a new DataFrame with only the desired columns
    df_train = processed_df[processed_df['split']=='train'].copy()
    df_test = processed_df[processed_df['split']=='test'].copy()
    
    prepare_and_save_json(df_train, dataset, output_train_json)
    prepare_and_save_json(df_test, dataset, output_test_json)

