FOR /L %%i IN (1, 1, 5) DO python -m src.training.experiment_conventional_multiclass ^
-p .\data\fromraw_cid_inchi_smiles_fp_labels_onehots.csv ^
--log-path .\logs\convention\test_1 ^
--max-depth 14 ^
--n-estimators 500 ^
--rand-seed %%i
