python -m src.training.experiment_linear_model_mixup ^
-p .\data\fromraw_cid_inchi_smiles_fp_labels_onehots.csv ^
-e 200 ^
-b 128 ^
--es-patience 10 ^
--log-path .\logs\linear ^
--repeat 3 ^
--mixup 0.4 ^
--mixup-repeat 10
