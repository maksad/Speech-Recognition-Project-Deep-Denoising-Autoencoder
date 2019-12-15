Base models:
    spec_base_model
    mel_base_model
    mfcc_base_model

    params:
        Total training files:  12548


commands:
    base models:
        python ./main.py --epochs 30 --training_data_dir ./clean_data_as_training --h5_dir_name ./clean_data_as_training/training_files_h5 --feat spec --num_cpu 10 --model_dir ./final_trained_models/spec_base_model --enhanced_dir ./data/enanced_data_final/spec_base_model
