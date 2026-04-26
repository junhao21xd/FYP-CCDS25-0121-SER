import pandas as pd

def combine_file_before_feature_extraction(original_csv, gender_pred_csv, vad_pred_csv, output_csv):
    # Load everything
    df_original = pd.read_csv(original_csv)
    df_gender = pd.read_csv(gender_pred_csv)
    df_vad = pd.read_csv(vad_pred_csv)

    final_df = df_original.merge(df_gender, on='id', how='left')
    final_df = final_df.merge(df_vad[['id', 'pred_valence', 'pred_arousal', 'pred_dominance']], on='id', how='left')

    # Save the ultimate dataset
    final_df.to_csv(output_csv, index=False)