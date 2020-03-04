python -m src.training.experiment_two_models ^
-p .\data\fromraw_cid_inchi_smiles_fp_labels_onehots_twoclass.csv ^
-e 200 ^
-b 128 ^
--es-patience 10 ^
--log-path .\logs\linear\two_modles ^
--repeat 1 ^
--rand-seed 241234
