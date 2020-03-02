python -m src.training.experiment_linear_balance_classes ^
-p .\data\fromraw_cid_inchi_smiles_fp_labels_onehots.csv ^
--outside-path .\data\DrugBank_smiles_fp.csv ^
-e 3 ^
-b 128 ^
--es-patience 10 ^
--log-path .\logs\linear\balanced ^
--repeat 1
