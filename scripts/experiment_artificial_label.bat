python -m src.training.experiment_artificial_label ^
-p .\data\fromraw_cid_inchi_smiles_fp_labels.csv ^
-e 1 ^
-b 256 ^
-l 0.001 ^
--es-patience 5 ^
--unlabeled-weight 0.5