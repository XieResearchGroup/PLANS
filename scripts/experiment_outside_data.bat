python -m src.training.experiment_outside_data ^
-p .\data\fromraw_cid_inchi_smiles_fp_labels.csv ^
-e 100 ^
-b 256 ^
-l 0.00005 ^
--es-patience 10 ^
--unlabeled-weight 1.0 ^
--log-path .\logs\outside_unlabeled\test_%1
