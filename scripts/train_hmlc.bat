@echo off
FOR /L %%i IN (1, 1, 5) DO ^
python -m src.training.train_hmlc ^
-p .\data\fromraw_cid_inchi_smiles_fp_labels.csv ^
-e 1000 ^
-b 256 ^
-l 0.00001 ^
--es-patience 10 ^
--unlabeled-weight 1.0 ^
--log-path .\logs\hmlc
