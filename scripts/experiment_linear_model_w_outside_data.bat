python -m src.training.experiment_linear_model_w_outside_data ^
-p .\data\fromraw_cid_inchi_smiles_fp_labels_onehots.csv ^
--outside-path .\data\DrugBank_smiles_fp.csv ^
-e 1 ^
-b 256 ^
--es-patience 10 ^
--log-path .\logs\linear ^
--repeat 1