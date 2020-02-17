@echo off
FOR /L %%i IN (1, 1, 5) DO ^
python -m src.training.experiment_M_L_wo_ns ^
-p .\data\fromraw_cid_inchi_smiles_fp_labels.csv ^
-e 1000 ^
-b 256 ^
-l 0.00005 ^
--es-patience 10 ^
--log-path .\logs\ML_w_labeled_only
