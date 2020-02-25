python -m src.training.experiment_linear_drugbank_mixup ^
-p .\data\fromraw_cid_inchi_smiles_fp_labels_onehots.csv ^
--outside-path .\data\DrugBank_smiles_fp.csv ^
-e 1000 ^
-b 128 ^
--es-patience 10 ^
--log-path .\logs\linear ^
--repeat 3 ^
--mixup 0.4 ^
--mixup-repeat 10