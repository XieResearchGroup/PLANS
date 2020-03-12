python -m src.training.experiment_linear_opt_mixup ^
-p .\data\fromraw_cid_inchi_smiles_fp_labels_onehots.csv ^
-e 2 ^
-b 128 ^
--es-patience 10 ^
--log-path .\logs\linear\opt_mixup ^
--repeat 1 ^
--mixup 0.4 ^
--mixup-repeat 10