python -m training.train_hmlc ^
-p ..\data\fromraw_cid_inchi_smiles_fp_labels.csv ^
-e 1 ^
-b 256 ^
-l 0.001 ^
--es-patience 5 ^
-m "train hlmc with weights" ^
--unlabeled-weight 0.5
