python -m src.training.experiment_mixup ^
-p .\data\fromraw_cid_inchi_smiles_fp_labels.csv ^
-e 200 ^
-b 256 ^
-l 0.0005 ^
--es-patience 100 ^
--unlabeled-weight 1 ^
--log-path .\logs\mixup
