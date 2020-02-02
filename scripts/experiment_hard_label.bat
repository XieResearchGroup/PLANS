python -m training.experiment_hard_label ^
-p ..\data\fromraw_cid_inchi_smiles_fp_labels.csv ^
-e 100 ^
-b 256 ^
-l 0.00001 ^
--es-patience 10 ^
--log-path ..\logs\soft_hard_labeling
