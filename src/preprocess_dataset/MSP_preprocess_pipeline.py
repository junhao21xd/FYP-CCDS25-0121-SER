from datasets import load_dataset, Dataset, ClassLabel, concatenate_datasets, Audio
from collections import defaultdict
import os
import soundfile as sf
import pandas as pd
import re
from tqdm import tqdm
import io

'''
Load MSP-Podcast dataset (subset of the original copy) from HuggingFace.
Recreate the audio array into wav files and save the path and metadata in csv. 
'''

# Load full dataset
ds = load_dataset("AbstractTTS/PODCAST", split="train")
ds = ds.shuffle(seed=42)

print("msp successfully downloaded")
wav_dir = os.path.abspath("../data/msp_dataset/wav_outputs")
os.makedirs(wav_dir, exist_ok=True)

output_csv = "../data/msp_dataset/msp_dataset_original.csv"

## Only works if ffmpeg is installed
#def compute_duration(example):
#    audio = example['audio']
#    audio_array = audio['array']
#    sampling_rate = audio['sampling_rate']
#    duration = len(audio_array) / sampling_rate
#    return {"duration": duration}

#def add_path(example):
#    audio = example['audio']
#    array = audio['array']
#    sampling_rate = audio['sampling_rate']
#    filename = example['id']
#    filepath = os.path.join(wav_dir, filename)
#    sf.write(filepath, array, sampling_rate)
#    return {"path": filepath}

#ds = ds.map(add_path)

def compute_duration(batch):
    durations = []
    
    for audio_data in batch['audio']:    
        # Scenario B: No path, but we have the raw file bytes
        if audio_data.get('bytes'):
            # Wrap the raw bytes in io.BytesIO so soundfile can read the headers
            virtual_file = io.BytesIO(audio_data['bytes'])
            info = sf.info(virtual_file)
            durations.append(info.duration)
            
        # Scenario C: Something is broken/missing in this row
        else:
            raise ValueError(f"Found a row with missing audio path and bytes! Row contents: {audio_data}")

    return {"duration": durations}

def add_path(example):
    # 2. Grab the raw file bytes instead of the decoded array
    audio_bytes = example['audio']['bytes']
    
    # 3. Create your filename and path
    # (Note: make sure your 'id' column actually exists!)
    filename = example['id'] 
    filepath = os.path.join(wav_dir, filename)
    
    # 4. Open a new file in "Write Binary" ('wb') mode and dump the bytes
    if audio_bytes is not None:
        with open(filepath, 'wb') as f:
            f.write(audio_bytes)
    else:
        raise ValueError(f"No audio bytes found for ID {example['id']}")
        
    # Return the new path so it gets saved in your dataset column
    return {"path": filepath}

ds = ds.cast_column("audio", Audio(decode=False))

# Apply to the dataset
ds = ds.map(compute_duration, batched=True, batch_size=1000)

# Initialize counters and containers
emotion_indices = defaultdict(list)
emotion_durations = defaultdict(float)  # in seconds

TARGET_DURATION = 3600  # 1 hour

# Collect up to 1 hour per emotion
for i, row in tqdm(enumerate(ds), total=len(ds)):
    emotion = row['major_emotion']

    if emotion_durations[emotion] < TARGET_DURATION:
        emotion_durations[emotion] += row['duration']
        emotion_indices[emotion].append(i)

all_indices = [idx for indices in emotion_indices.values() for idx in indices]
ds = ds.select(all_indices)

ds = ds.rename_column(
    "major_emotion", "emotion"
)

# dup the col with original col name
ds = ds.add_column(
    "major_emotion", ds["emotion"]
)

ds = ds.rename_column(
    "EmoVal", "valence"
)

ds = ds.rename_column(
    "EmoAct", "arousal"
)

ds = ds.rename_column(
    "EmoDom", "dominance"
)

ds = ds.rename_column(
    "file", "id"
)

ds = ds.rename_column(
    "transcription", "text"
)

unique_emotions = sorted(set(ds["emotion"]))
class_label = ClassLabel(names=unique_emotions)

# cast to allow stratify split
ds = ds.cast_column(
    "major_emotion", class_label
)

split_dataset = ds.train_test_split(
    test_size=0.2,
    stratify_by_column="major_emotion",
    seed=42
)

train_ds = split_dataset['train'].add_column("split", ["train"] * len(split_dataset['train']))
test_ds = split_dataset['test'].add_column("split", ["test"] * len(split_dataset['test']))

ds = concatenate_datasets([train_ds, test_ds])

ds = ds.map(add_path)

ds = ds.remove_columns("audio")

df = ds.to_pandas()

def clean_text(text):
    if not isinstance(text, str):
        return ""
    
    # 1. Fix formatting: "noise] actual_text [" → keep only the "actual_text"
    match = re.search(r'\]\s*(.*?)\s*\[', text)
    if match:
        text = match.group(1)

    # 2. Replace [crosstalk ...] and [inaudible ...] with placeholders
    text = re.sub(r'\[crosstalk[^\]]*?\]', '(crosstalk)', text, flags=re.IGNORECASE)
    text = re.sub(r'\[inaudible[^\]]*?\]', '(inaudible)', text, flags=re.IGNORECASE)
    text = re.sub(r'\[laughter[^\]]*?\]', '(laughter)', text, flags=re.IGNORECASE)
    text = re.sub(r'\[foreign language[^\]]*?\]', '(inaudible)', text, flags=re.IGNORECASE)
    text = re.sub(r'\[in audible[^\]]*?\]', '(inaudible)', text, flags=re.IGNORECASE)
    text = re.sub(r'\[german[^\]]*?\]', '(inaudible)', text, flags=re.IGNORECASE)

    # 3. Replace patterns like [Tom 00:02:24] → Tom
    text = re.sub(
        r'\[([^\[\]]*?)\s+\d\d:\d\d(?::\d\d)?(?:\.\d+)?\]',  # e.g., [Tom 00:02:24]
        r'\1',
        text
    )
    
    # 4. Remove timestamp: [00:02:24]
    text = re.sub(r'\[\s*\d{1,2}(:\d{2}){1,2}(\.\d+)?\s*\]', '', text)

    # 5. Remove brackets around remaining content, preserving inner text
    text = re.sub(r'\[([^\[\]]+)\]', r'\1', text)

    # 6. Remove any unmatched brackets (e.g., stray [ or ])
    text = text.replace('[', '').replace(']', '')

    # Remove specific label tags (case insensitive)
    return text.strip()

df = df.dropna(subset=['text'])

df['text'] = df['text'].apply(lambda x: clean_text(x))
df = df[df['text'] != ""]

cols = [
        "gender", "emotion", "valence", "arousal", "dominance", 
        "duration", "path", "text", "id", "split"
    ]
df[cols].to_csv(output_csv,index=False)
