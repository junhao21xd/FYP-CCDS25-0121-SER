import os
import re
import glob
import pandas as pd
import numpy as np
import soundfile as sf  

# --- CONFIGURATION ---
IEMOCAP_TRANSCRIPTION_PATH = "../data/iemocap_dataset/IEMOCAP_full_release/Session*/dialog/transcriptions" 

IEMOCAP_ROOT_PATH = "../data/iemocap_dataset/IEMOCAP_full_release"
OUTPUT_CSV_PATH = "../data/iemocap_dataset/iemocap_dataset_original.csv"

def parse_iemocap(root_path):
    data = []
    
    # Regex pattern matches: [TIME] ID EMOTION [V, A, D]
    pattern = re.compile(r"\[.*?\]\s+(Ses\w+)\s+([a-z]{3})\s+\[(\s*-?[0-9\.]+\s*,\s*-?[0-9\.]+\s*,\s*-?[0-9\.]+\s*)\]")
    
    for session_id in range(1, 6):
        session_name = f'Session{session_id}'
        print(f"Processing {session_name}...")
        
        base_path = os.path.join(root_path, session_name)
        wav_root = os.path.join(base_path, 'sentences', 'wav')
        emo_eval_root = os.path.join(base_path, 'dialog', 'EmoEvaluation')
        trans_root = os.path.join(base_path, 'dialog', 'transcriptions')
        
        if not os.path.isdir(base_path):
            print(f"  -> Skipping {session_name} (folder not found)")
            continue

        # --- 1. Parse EmoEvaluation (Emotion + VAD) ---
        metadata_map = {} 
        
        emo_files = glob.glob(os.path.join(emo_eval_root, '*.txt'))
        for emo_file in emo_files:
            # Skip macOS '._' hidden files
            if os.path.basename(emo_file).startswith('._'):
                continue

            try:
                with open(emo_file, 'r', encoding='utf-8', errors='ignore') as f:
                    content = f.read()
            except Exception as e:
                print(f"  -> Error reading file {os.path.basename(emo_file)}: {e}")
                continue
            
            for line in content.splitlines():
                if not line.strip().startswith('['):
                    continue
                    
                match = pattern.search(line)
                if match:
                    utt_id = match.group(1)
                    emotion = match.group(2)
                    vad_str = match.group(3)
                    
                    try:
                        val, act, dom = [float(x) for x in vad_str.split(',')]
                        metadata_map[utt_id] = {
                            'emotion': emotion,
                            'valence': val,
                            'arousal': act,
                            'dominance': dom,
                            'session': session_id
                        }
                    except ValueError:
                        continue

        # --- 2. Parse Transcriptions (Text) ---
        transcript_map = {}
        trans_files = glob.glob(os.path.join(trans_root, '*.txt'))
        
        for trans_file in trans_files:
            if os.path.basename(trans_file).startswith('._'):
                continue
            
            try:
                with open(trans_file, 'r', encoding='utf-8', errors='ignore') as f:
                    lines = f.readlines()
            except Exception as e:
                pass 

            for line in lines:
                parts = line.strip().split(']: ')
                if len(parts) >= 2:
                    meta = parts[0].split(' ')[0]
                    if 'Ses' in meta:
                        transcript_map[meta] = parts[1]

        # --- 3. Combine Data & Get Duration ---
        for utt_id, meta in metadata_map.items():
            
            parts = utt_id.split('_')
            video_id = "_".join(parts[:-1])
            suffix = parts[-1]
            gender_code = suffix[0] 
            
            try:
                segment_id = int(suffix[1:])
            except ValueError:
                segment_id = -1
                
            text = transcript_map.get(utt_id, "")
            
            wav_path = os.path.join(wav_root, video_id, f"{utt_id}.wav")
            
            if os.path.exists(wav_path):
                duration = 0.0
                try:
                    audio_info = sf.info(wav_path)
                    duration = audio_info.duration
                except Exception as e:
                    print(f"  -> Warning: Could not read duration for {utt_id}: {e}")
                    duration = 0.0

                data.append({
                    "gender": gender_code,
                    "emotion": meta['emotion'],
                    "valence": meta['valence'],
                    "arousal": meta['arousal'],
                    "dominance": meta['dominance'],
                    "duration": duration,
                    "session": meta['session'],
                    "path": os.path.abspath(wav_path),
                    "text": text,
                    "id": utt_id,
                    "segment_id": segment_id,
                    "video_id": video_id
                })

    return pd.DataFrame(data)

def create_order_index(iemocap_root_path):
    """
    Parses IEMOCAP transcriptions to create a chronologically sorted index based on conversation files.
    """
    data = []
    transcription_dirs = glob.glob(iemocap_root_path)

    for tdir in transcription_dirs: 
    # Walk through the dataset structure
    # Expected structure: SessionX/dialog/transcriptions/
        for root, dirs, files in os.walk(tdir):
            for file in files:
                if file.endswith(".txt") and not file.startswith("."):
                    
                    # The filename (minus .txt) is usually the Dialogue_ID
                    dialogue_id = os.path.splitext(file)[0]
                    file_path = os.path.join(root, file)
                    
                    # We collect all turns for this specific dialogue here
                    dialogue_turns = []
                    
                    with open(file_path, 'r') as f:
                        for line in f:
                            # Regex to find: UtteranceID [Start-End]:
                            # Matches: Ses01F_impro01_F000 [6.2900-8.2350]:
                            match = re.match(r"(Ses\w+)\s+\[(\d+\.\d+)-(\d+\.\d+)\]:", line)
                            
                            if match:
                                utt_id = match.group(1)
                                start_time = float(match.group(2))
                                
                                dialogue_turns.append({
                                    "video_id": dialogue_id,
                                    "id": utt_id,
                                    "Start_Time": start_time
                                })
                    
                    # SORTING LOGIC:
                    # 1. Sort this specific dialogue by Start_Time
                    dialogue_turns.sort(key=lambda x: x["Start_Time"])
                    
                    # 2. Add the reset index (0, 1, 2...)
                    for idx, turn in enumerate(dialogue_turns):
                        turn["Order_Index"] = idx
                        data.append(turn)

    return pd.DataFrame(data)


if __name__ == "__main__":
    
    os.makedirs(IEMOCAP_ROOT_PATH, exist_ok=True)
    df = parse_iemocap(IEMOCAP_ROOT_PATH)
    
    cols = [
        "gender", "emotion", "valence", "arousal", "dominance", 
        "duration", "path", "text", "id", "segment_id", "video_id",
        "session", "split"
    ]
    df['split'] = np.where(df['session'].between(1, 4), 'train', 'test')
    df = df[cols]
    
    df_order = create_order_index(IEMOCAP_TRANSCRIPTION_PATH)
    df_final = pd.merge(df, df_order[['id', 'Order_Index']], 
                on='id', 
                how='left')

    # Sort by Dialogue ID first, then by the new Order Index
    df_final = df_final.sort_values(by=['video_id', 'Order_Index'])

    df_final.to_csv(OUTPUT_CSV_PATH, index=False)
    print(f"Saved to {OUTPUT_CSV_PATH}")
