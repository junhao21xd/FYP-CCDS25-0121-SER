
DATASET_PROMPT_GROUPS = {
    "msp": {
        "Pitch (F0)": ['Average Pitch', 'Pitch Stability (StdDev)'],
        "Loudness": [
            'Average Loudness', 'Overall Sound Level (dBP)', 'Loudness Range', 
            'Loudness Variation (StdDev)', 'Avg. Loudness Decrease Slope', 
            'Avg. Loudness Increase Slope', 'Loudness Peaks per Second', 
            'Loudness 20th Percentile', 'Loudness Decrease Variability'
        ],
        "Spectral / Formant Related": [
            'Spectral Slope (500-1500 Hz)', 'Spectral Flux (Timbre Change)',
            'Spectral Flux (Unvoiced Regions)', 'Spectral Flux Variation (Voiced)',
            'Alpha Ratio (Spectral Balance)', 'F1 Frequency (Avg)',
            'Hammarberg Index (Voice Sharpness)'
        ],
        "Pace & Timing": [
            'Speaking Rate', 'Avg. Unvoiced Length',
            'Unvoiced Length Variation', 'Voiced Length Variation (StdDev)'
        ],
        "Voice Quality": [
            'MFCC 1 (Spectral Shape)', 'Harmonic-Formant Diff (H1-A3)',
            'Jitter (Voice Roughness)', 'Shimmer (Voice Breathiness)'
        ],
    },
    
    "iemocap": {
        "Pitch (Intonation)": [
            'Average Pitch', 'Pitch Stability (StdDev)', 'Pitch Range (Dynamic Range)',
        ],

        "Loudness (Energy)": [
            'Average Loudness', 'Overall Sound Level (dBP)', 'Loudness Variation (StdDev)', 'Avg. Loudness Increase Slope', 'Loudness Decrease Variability',
        ],

        "Rhythm & Speed": [
            'Speaking Rate', 'Unvoiced Length Variation (Pause/Consonant Variability)',
        ],

        "Voice Quality (Roughness/Breathiness)": [
            'Jitter (Voice Roughness)', 'Shimmer (Voice Breathiness)', 'Harmonics-to-Noise Ratio (Voice Clarity)', 'Hammarberg Variation (Breathiness Stability)',
        ],

        "Spectral Balance (Timbre & Tone)": [
            'Alpha Ratio (Spectral Balance - Voiced)', 'Alpha Ratio (Spectral Balance - Unvoiced)', 'Low Freq Spectral Slope (Voiced)', 
            'Low Freq Spectral Slope (Unvoiced)', 'Overall Spectral Flux Variation', 'Spectral Flux Variation (Voiced)',
        ],

        "Formants (Vowel Articulation)": [
            'F1 Frequency (Vowel Openness)', 'F1 Amplitude (Relative to Pitch)',
        ],

        "MFCCs (Abstract Spectral Shape)": [
            'MFCC 1 (Overall Spectral Shape)', 'MFCC 1 (Voiced Spectral Shape)', 'MFCC 1 Variation (Overall)', 'MFCC 1 Variation (Voiced)', 
            'MFCC 2 (Voiced Spectral Energy)', 'MFCC 4 (Overall High-Freq Detail)', 'MFCC 4 (Voiced High-Freq Detail)',
        ],
    }
}


FEATURE_RENAMING_MAPS = {
    "msp": {
        # Pitch (F0)
        'F0semitoneFrom27.5Hz_sma3nz_amean': 'Average Pitch',
        'F0semitoneFrom27.5Hz_sma3nz_stddevNorm': 'Pitch Stability (StdDev)',
        
        # Loudness
        'loudness_sma3_amean': 'Average Loudness',
        'equivalentSoundLevel_dBp': 'Overall Sound Level (dBP)',
        'loudness_sma3_pctlrange0-2': 'Loudness Range',
        'loudness_sma3_stddevNorm': 'Loudness Variation (StdDev)',
        'loudness_sma3_meanFallingSlope': 'Avg. Loudness Decrease Slope',
        'loudness_sma3_meanRisingSlope': 'Avg. Loudness Increase Slope',
        'loudnessPeaksPerSec': 'Loudness Peaks per Second',
        'loudness_sma3_percentile20.0': 'Loudness 20th Percentile',
        'loudness_sma3_stddevFallingSlope': 'Loudness Decrease Variability',

        # Spectral / Formant Related
        'slopeV500-1500_sma3nz_amean': 'Spectral Slope (500-1500 Hz)',
        'spectralFluxV_sma3nz_amean': 'Spectral Flux (Timbre Change)',
        'spectralFluxUV_sma3nz_amean': 'Spectral Flux (Unvoiced Regions)',
        'spectralFluxV_sma3nz_stddevNorm': 'Spectral Flux Variation (Voiced)',
        'alphaRatioV_sma3nz_amean': 'Alpha Ratio (Spectral Balance)',
        'hammarbergIndexV_sma3nz_amean': 'Hammarberg Index (Voice Sharpness)',

        # Pace & Timing
        'VoicedSegmentsPerSec': 'Speaking Rate',
        'MeanUnvoicedSegmentLength': 'Avg. Unvoiced Length',
        'StddevUnvoicedSegmentLength': 'Unvoiced Length Variation',
        'StddevVoicedSegmentLengthSec': 'Voiced Length Variation (StdDev)',

        # Voice Quality
        'mfcc1V_sma3nz_amean': 'MFCC 1 (Spectral Shape)',
        'logRelF0-H1-A3_sma3nz_amean': 'Harmonic-Formant Diff (H1-A3)',
        'jitterLocal_sma3nz_amean': 'Jitter (Voice Roughness)',
        'shimmerLocaldB_sma3nz_amean': 'Shimmer (Voice Breathiness)',

        # Formants (Vowels)
        'F1frequency_sma3nz_amean': 'F1 Frequency (Avg)',
    },
    
    "iemocap": {
        # --- Pitch (Intonation) ---
        'F0semitoneFrom27.5Hz_sma3nz_amean': 'Average Pitch',
        'F0semitoneFrom27.5Hz_sma3nz_stddevNorm': 'Pitch Stability (StdDev)',
        'F0semitoneFrom27.5Hz_sma3nz_pctlrange0-2': 'Pitch Range (Dynamic Range)',

        # --- Loudness (Energy) ---
        'loudness_sma3_amean': 'Average Loudness',
        'equivalentSoundLevel_dBp': 'Overall Sound Level (dBP)',
        'loudness_sma3_stddevNorm': 'Loudness Variation (StdDev)',
        'loudness_sma3_meanRisingSlope': 'Avg. Loudness Increase Slope',
        'loudness_sma3_stddevFallingSlope': 'Loudness Decrease Variability',

        # --- Rhythm & Speed ---
        'VoicedSegmentsPerSec': 'Speaking Rate',
        'StddevUnvoicedSegmentLength': 'Unvoiced Length Variation (Pause/Consonant Variability)',

        # --- Voice Quality (Roughness/Breathiness) ---
        'jitterLocal_sma3nz_amean': 'Jitter (Voice Roughness)',
        'shimmerLocaldB_sma3nz_amean': 'Shimmer (Voice Breathiness)',
        'HNRdBACF_sma3nz_amean': 'Harmonics-to-Noise Ratio (Voice Clarity)',
        'hammarbergIndexV_sma3nz_stddevNorm': 'Hammarberg Variation (Breathiness Stability)',

        # --- Spectral Balance (Timbre & Tone) ---
        'alphaRatioV_sma3nz_amean': 'Alpha Ratio (Spectral Balance - Voiced)',
        'alphaRatioUV_sma3nz_amean': 'Alpha Ratio (Spectral Balance - Unvoiced)',
        'slopeV0-500_sma3nz_amean': 'Low Freq Spectral Slope (Voiced)',
        'slopeUV0-500_sma3nz_amean': 'Low Freq Spectral Slope (Unvoiced)',
        'spectralFlux_sma3_stddevNorm': 'Overall Spectral Flux Variation',
        'spectralFluxV_sma3nz_stddevNorm': 'Spectral Flux Variation (Voiced)',

        # --- Formants (Vowel Articulation) ---
        'F1frequency_sma3nz_amean': 'F1 Frequency (Vowel Openness)',
        'F1amplitudeLogRelF0_sma3nz_amean': 'F1 Amplitude (Relative to Pitch)',

        # --- MFCCs (Abstract Spectral Shape) ---
        # MFCC 1 usually correlates with spectral tilt/balance
        'mfcc1_sma3_amean': 'MFCC 1 (Overall Spectral Shape)',
        'mfcc1V_sma3nz_amean': 'MFCC 1 (Voiced Spectral Shape)',
        'mfcc1_sma3_stddevNorm': 'MFCC 1 Variation (Overall)',
        'mfcc1V_sma3nz_stddevNorm': 'MFCC 1 Variation (Voiced)',
        
        # MFCC 2 usually correlates with spectral energy distribution
        'mfcc2V_sma3nz_amean': 'MFCC 2 (Voiced Spectral Energy)',
        
        # Higher MFCCs (3-4) correlate with finer spectral details
        'mfcc4_sma3_amean': 'MFCC 4 (Overall High-Freq Detail)',
        'mfcc4V_sma3nz_amean': 'MFCC 4 (Voiced High-Freq Detail)',
    }

}

EXTRACTED_FEATURE_SET = {
    'msp': ['Average Pitch', 'Pitch Stability (StdDev)', 'Average Loudness', 'Overall Sound Level (dBP)', 'Loudness Range', 'Loudness Variation (StdDev)', 'Avg. Loudness Decrease Slope', 'Avg. Loudness Increase Slope', 'Loudness Peaks per Second', 'Loudness 20th Percentile', 'Loudness Decrease Variability', 'Spectral Slope (500-1500 Hz)', 'Spectral Flux (Timbre Change)', 'Spectral Flux (Unvoiced Regions)', 'Spectral Flux Variation (Voiced)', 'Alpha Ratio (Spectral Balance)', 'Hammarberg Index (Voice Sharpness)', 'Speaking Rate', 'Avg. Unvoiced Length', 'Unvoiced Length Variation', 'Voiced Length Variation (StdDev)', 'MFCC 1 (Spectral Shape)', 'Harmonic-Formant Diff (H1-A3)', 'Jitter (Voice Roughness)', 'Shimmer (Voice Breathiness)', 'F1 Frequency (Avg)'],
    
    'iemocap': ['Average Pitch', 'Pitch Stability (StdDev)', 'Pitch Range (Dynamic Range)', 'Average Loudness', 'Overall Sound Level (dBP)', 'Loudness Variation (StdDev)', 'Avg. Loudness Increase Slope', 'Loudness Decrease Variability', 'Speaking Rate', 'Unvoiced Length Variation (Pause/Consonant Variability)', 'Jitter (Voice Roughness)', 'Shimmer (Voice Breathiness)', 'Harmonics-to-Noise Ratio (Voice Clarity)', 'Hammarberg Variation (Breathiness Stability)', 'Alpha Ratio (Spectral Balance - Voiced)', 'Alpha Ratio (Spectral Balance - Unvoiced)', 'Low Freq Spectral Slope (Voiced)', 'Low Freq Spectral Slope (Unvoiced)', 'Overall Spectral Flux Variation', 'Spectral Flux Variation (Voiced)', 'F1 Frequency (Vowel Openness)', 'F1 Amplitude (Relative to Pitch)', 'MFCC 1 (Overall Spectral Shape)', 'MFCC 1 (Voiced Spectral Shape)', 'MFCC 1 Variation (Overall)', 'MFCC 1 Variation (Voiced)', 'MFCC 2 (Voiced Spectral Energy)', 'MFCC 4 (Overall High-Freq Detail)', 'MFCC 4 (Voiced High-Freq Detail)', 'valence','arousal','dominance']
}

GENDER_CLASSIFIER_CONFIGS = {
    "iemocap": {
        "eval_strategy":"epoch",
        "save_strategy":"epoch",
        "logging_steps":10,
        "learning_rate":3e-5,            
        "per_device_train_batch_size":8, 
        "per_device_eval_batch_size":8,
        "num_train_epochs":5,
        "load_best_model_at_end":True,
        "metric_for_best_model":"accuracy",
        "save_total_limit":1,
        "gradient_checkpointing":True
    },
    
    "msp": {
        "eval_strategy":"epoch",
        "save_strategy":"epoch",
        "logging_steps":10,
        "learning_rate":3e-5,            
        "per_device_train_batch_size":8, 
        "per_device_eval_batch_size":8,
        "num_train_epochs":5,
        "load_best_model_at_end":True,
        "metric_for_best_model":"accuracy",
        "save_total_limit":1,
        "gradient_checkpointing":True,
        "max_grad_norm":1.0,       
        "warmup_ratio":0.1,     
        "weight_decay":0.01,
    },
    
    "default": {
        "eval_strategy":"epoch",
        "save_strategy":"epoch",
        "logging_steps":10,
        "learning_rate":1e-5,            
        "per_device_train_batch_size":8, 
        "per_device_eval_batch_size":8,
        "num_train_epochs":5,
        "load_best_model_at_end":True,
        "metric_for_best_model":"accuracy",
        "save_total_limit":1,
        "gradient_checkpointing":True,
        "max_grad_norm":1.0,       
        "warmup_ratio":0.1,     
        "weight_decay":0.01,
    }
}
