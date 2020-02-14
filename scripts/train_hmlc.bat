python -m src.training.train_hmlc ^
-p .\data\fromraw_cid_inchi_smiles_fp_labels.csv ^
-e 1 ^
-b 256 ^
-l 0.001 ^
--es-patience 5 ^
--unlabeled-weight 0.5
--log-path .\logs\hmlc
