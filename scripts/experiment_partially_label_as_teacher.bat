FOR %%i IN (1, 1, 5) DO ^
python -m src.training.experiment_partially_label_as_teacher ^
-p .\data\fromraw_cid_inchi_smiles_fp_labels.csv ^
-e 1000 ^
-b 256 ^
-l 0.00005 ^
--es-patience 10 ^
--unlabeled-weight 1.0 ^
--log-path .\logs\partially_label_as_teacher
