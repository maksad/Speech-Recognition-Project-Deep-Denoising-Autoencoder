# Models
    Base models:
        spec_base_model:                    Loss Reg: 0.0186
        mel_base_model:                     Loss Reg: 0.0292
        mfcc_base_model:                    Loss Reg: 0.3222

        params:
            Total training files:  12548

    Denoising model
        spec_denoise_model
            start without base model:        Loss Reg: 1.8046
            start with base model:           Loss Reg: 1.5358
            final loss with base:            1.02230
        mel_denoise_model
            start without base model:        Loss Reg: 6.3143
            start with base model:           Loss Reg: 4.7896
            final loss with base:            2.3266
        mfcc_denoise_model
            start without base model:        Loss Reg: 505.2360
            start with base model:           Loss Reg: 394.2066

        params:
            Total training files:  2977


# commands:
    base models:
        python ./main.py --epochs 30 --training_data_dir ./clean_data_as_training --h5_dir_name ./clean_data_as_training/training_files_h5 --feat spec --num_cpu 10 --model_dir ./final_trained_models/spec_base_model --enhanced_dir ./data/enanced_data_final/spec_base_model

    train on base models:
        python ./main.py --epochs 300 --feat spec --num_cpu 10 --model_dir ./final_trained_models/spec_denoising_model --enhanced_dir ./data/enanced_data_final/spec_denoising_model --base_model_dir ./final_trained_models/spec_base_model/

    test base model:
        python main.py --train false --testing_data_dir ./data/test_data/clean --enhanced_dir ./data/enanced_data_final/mel_base_model --model_dir ./final_trained_models/mel_base_model
