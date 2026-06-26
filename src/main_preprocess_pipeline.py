import argparse
from audio_preprocess_model import (
    run_asr, run_gender_classifier_evaluation, 
    run_gender_classifier_inference, train_gender_classifier, 
    preprocess_iemocap_audeering, preprocess_msp_audeering, train_audeering, evaluate_audeering, inference_audeering,
    extract_feature_opensmile, process_audio_feature, combine_file_before_feature_extraction
    )

def main():
    parser = argparse.ArgumentParser(description="Master Audio Pipeline")
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--original_csv")
    
    parser.add_argument("--feature_extractor_path", default="facebook/wav2vec2-base")
    parser.add_argument("--gender_classifier_path", default="facebook/wav2vec2-base")
    parser.add_argument("--train_gender_classifier", action="store_true")
    parser.add_argument("--eval_gender_classifier", action="store_true")
    parser.add_argument("--inference_gender_classifier", action="store_true")
    parser.add_argument("--gender_input_csv", type=str, help="input audio file to classify gender")
    parser.add_argument("--gender_output_csv", type=str, help="output audio file that stores gender prediction and probability")
    
    parser.add_argument("--vad_feature_extractor_path", default="audeering/wav2vec2-large-robust-12-ft-emotion-msp-dim")
    parser.add_argument("--vad_regressor_path", default="audeering/wav2vec2-large-robust-12-ft-emotion-msp-dim")
    parser.add_argument("--preprocess_vad_data", action="store_true")
    parser.add_argument("--generate_oof_prep", action="store_true", help="whether to prep 5 fold of train-test data")
    parser.add_argument("--train_vad_regressor", action="store_true")
    parser.add_argument("--generate_oof_train", action="store_true", help="whether to train 5 models to obtain the pred of each eval fold")
    parser.add_argument("--eval_vad_regressor", action="store_true")
    parser.add_argument("--inference_vad_regressor", action="store_true")
    parser.add_argument("--generate_oof_eval", action="store_true", help="whether to eval/inference get Out-of-Fold (OOF) predictions")
    parser.add_argument("--vad_input_csv", type=str, help="input audio file to regress vad")
    parser.add_argument("--vad_output_path", type=str, help="path that stores vad prediction output file")
    
    parser.add_argument("--combine_files", action="store_true")
    parser.add_argument("--combined_output_csv", type=str)
    
    parser.add_argument("--run_extractor", action="store_true")
    parser.add_argument("--feature_extraction_input_csv", type=str)
    parser.add_argument("--process_audio_feature", action="store_true")
    parser.add_argument("--use_pred_gender", action="store_true", help="options to calculate threshold with predicted gender instead of ground truth.")
    parser.add_argument("--input_audio_folder", type=str, help="input audio folder that contains wav file to extract audio feature")
    parser.add_argument("--feature_input_csv", type=str, help="csv file to store extracted feature output file")
    parser.add_argument("--feature_output_path", type=str, help="path that stores extracted feature output file into train and test json")
    
    parser.add_argument("--run_asr", action="store_true")
    parser.add_argument("--asr_model_id", default="openai/whisper-large-v3-turbo")
    parser.add_argument("--calc_wer", action="store_true")
    parser.add_argument("--normalize", action="store_true")
    parser.add_argument("--asr_train_input_json", type=str, help="input audio file to run asr on")
    parser.add_argument("--asr_train_output_json", type=str, help="output audio file that stores asr transcription")
    parser.add_argument("--asr_test_input_json", type=str, help="input audio file to run asr on")
    parser.add_argument("--asr_test_output_json", type=str, help="output audio file that stores asr transcription")
    
    args = parser.parse_args()
    
    ## Gender Classifier
    if args.train_gender_classifier:
        print("Training Gender Classifier")
        finetuned_model_path = train_gender_classifier(args.dataset, args.gender_input_csv, args.feature_extractor_path, args.gender_classifier_path)
    else:
        finetuned_model_path = args.gender_classifier_path
    
    if args.eval_gender_classifier:
        print("Evaluating Gender Classifier")
        run_gender_classifier_evaluation(args.gender_input_csv, args.feature_extractor_path, finetuned_model_path, args.gender_output_csv)
    elif args.inference_gender_classifier:
        print("Evaluating Gender Classifier")
        run_gender_classifier_inference(args.gender_input_csv, args.feature_extractor_path, finetuned_model_path, args.gender_output_csv)
    
    
    ## VAD regressor
    if args.preprocess_vad_data:
        if args.dataset == 'msp':
            print("Preprocessing data for VAD Regressor")
            preprocess_msp_audeering(args.vad_input_csv, args.vad_output_path, args.generate_oof_prep)
        elif args.dataset == 'iemocap':
            print("Preprocessing data for VAD Regressor")
            preprocess_iemocap_audeering(args.vad_input_csv, args.vad_output_path, args.generate_oof_prep)
    
    if args.train_vad_regressor:
        print("Training VAD models")
        train_audeering(args.vad_regressor_path, args.vad_output_path, args.dataset, args.generate_oof_train)
    
    if args.eval_vad_regressor:
        print("Evaluating VAD")
        vad_output_csv = evaluate_audeering(args.vad_feature_extractor_path, args.vad_output_path, args.dataset, args.generate_oof_eval)
    elif args.inference_vad_regressor:
        print("Evaluating VAD")
        vad_output_csv = inference_audeering(args.vad_feature_extractor_path, args.vad_output_path, args.dataset, args.generate_oof_eval)

    else:
        vad_output_csv = f"{args.vad_output_path}/{args.dataset}_vad_eval_predictions.csv"
        
    ## Combine files
    if args.combine_files:
        print("Combining Files")
        combine_file_before_feature_extraction(args.original_csv, args.gender_output_csv, vad_output_csv, args.combined_output_csv)
        
    ## eGeMaps Feature Extractor
    if args.run_extractor:
        print("Extracting eGeMaps Feature from audio")
        feature_extraction_output_csv = extract_feature_opensmile(args.dataset, args.feature_extraction_input_csv)

    else:
        feature_extraction_output_csv = f"../data/{args.dataset}_dataset/{args.dataset}_dataset_egemaps_features.csv"
        
    if args.process_audio_feature:
        print("Processing eGeMaps")
        process_audio_feature(args.dataset, feature_extraction_output_csv, args.feature_output_path)
        # process_audio_feature(args.dataset, feature_extraction_output_csv, args.feature_output_path, args.use_pred_gender)
    
    ## ASR
    if args.run_asr:
        print("Running ASR")
        run_asr(
            input_json=args.asr_test_input_json,
            output_json=args.asr_test_output_json,
            model_id=args.asr_model_id,
            calc_wer=args.calc_wer,
            normalize_wer=args.normalize
        )

if __name__ == "__main__":
    main()