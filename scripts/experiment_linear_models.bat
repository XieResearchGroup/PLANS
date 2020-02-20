python -m src.training.experiment_linear_models ^
-p .\data\fromraw_cid_inchi_smiles_fp_labels_onehots.csv ^
-e 200 ^
-b 256 ^
-l 0.000001 ^
--es-patience 10 ^
--log-path .\logs\linear