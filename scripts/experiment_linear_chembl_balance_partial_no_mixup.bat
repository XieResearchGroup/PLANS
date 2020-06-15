python -m src.training.experiment_linear_chembl_balance_partial_no_mixup ^
-p .\data\fromraw_cid_inchi_smiles_fp_labels_onehots.csv ^
--outside-path .\data\ChEMBL24_test.hdf5 ^
-e 1 ^
-b 64 ^
--es-patience 10 ^
--log-path .\logs\linear\chembl_balanced_partial_no_mixup ^
--repeat 1 ^
--rand-seed 0 ^
