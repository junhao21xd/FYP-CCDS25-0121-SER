import opensmile
import pandas as pd
import glob
import os


def extract_feature_opensmile(dataset, data_path, audio_folder=None):
    # Initialize eGeMAPS (extended version)
    smile = opensmile.Smile(
        feature_set=opensmile.FeatureSet.eGeMAPS,  # eGeMAPSv02
        feature_level=opensmile.FeatureLevel.Functionals
    )

    # Folder with your audio files
    if audio_folder is None:
        if dataset == 'iemocap':
            audio_folder = "../data/iemocap_dataset/IEMOCAP_full_release/Session*/sentences/wav/*/*.wav"
        elif dataset == 'msp':
            audio_folder = "../data/msp_dataset/wav_outpus/*.wav"


    data_df = pd.read_csv(data_path)

    # Store results
    all_features = []

    # Loop through each audio file
    for file in glob.glob(audio_folder):
        if file.endswith('.wav') and not file.startswith('._'):
            features = smile.process_file(file)
            features['id'] = os.path.basename(file)[:-4]  # Keep track of filename
            all_features.append(features)

    df = pd.concat(all_features)
    final_df = data_df.merge(df, on='id', how='inner')
    final_df.reset_index(drop=True, inplace=True)

    final_df.to_csv(f"../data/{dataset}_dataset/{dataset}_dataset_egemaps_features.csv", index=False)

    print("Features extracted and saved to egemaps_features.csv")