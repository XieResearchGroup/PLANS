python -m src.training.experiment_linear_chembl_balance ^
-p .\data\fromraw_cid_inchi_smiles_fp_labels_onehots.csv ^
--outside-path .\data\ChEMBL24.hdf5 ^
-e 1 ^
-b 128 ^
--es-patience 10 ^
--log-path .\logs\linear\chembl24_balanced ^
--repeat 1
