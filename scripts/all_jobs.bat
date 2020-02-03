@echo off 
set atempt=atempt_%1
REM REM Training S, M, L models with Noisy Student
FOR /L %%i IN (1, 1, 5) DO ^
python -m src.training.train_hmlc ^
-p .\data\fromraw_cid_inchi_smiles_fp_labels.csv ^
-e 1000 ^
-b 256 ^
-l 0.0001 ^
--es-patience 10 ^
--unlabeled-weight 1.0 ^
--log-path .\logs\%atempt%\NoisyStudent\test_%%i

REM REM Test M and L models with only the labeled data
FOR /L %%i IN (1, 1, 5) DO ^
python -m src.training.experiment_M_L_wo_ns ^
-p .\data\fromraw_cid_inchi_smiles_fp_labels.csv ^
-e 1000 ^
-b 256 ^
-l 0.0001 ^
--es-patience 10 ^
--log-path .\logs\%atempt%\ML_w_labeled_only\test_%%i ^
--unlabeled-weight 1.0

REM REM Test whether soft labeling is better than hard labeling
FOR /L %%i IN (1, 1, 5) DO ^
python -m src.training.experiment_hard_label ^
-p .\data\fromraw_cid_inchi_smiles_fp_labels.csv ^
-e 1000 ^
-b 256 ^
-l 0.0001 ^
--es-patience 10 ^
--log-path .\logs\%atempt%\soft_hard_labeling\test_%%i ^
--unlabeled-weight 1.0

REM REM Test adding loss weights
FOR /L %%i IN (1, 1, 5) DO ^
python -m src.training.experiment_exploit_unlabeled ^
-p .\data\fromraw_cid_inchi_smiles_fp_labels.csv ^
-e 1000 ^
-b 256 ^
-l 0.0001 ^
--es-patience 10 ^
--log-path .\logs\%atempt%\loss_w_weights\test_%%i

REM REM Test artificial labeling
FOR /L %%i IN (1, 1, 5) DO ^
python -m src.training.experiment_artificial_label ^
-p .\data\fromraw_cid_inchi_smiles_fp_labels.csv ^
-e 1000 ^
-b 256 ^
-l 0.0001 ^
--es-patience 10 ^
--unlabeled-weight 1.0 ^
--log-path .\logs\%atempt%\artificial_labeling\test_%%i

REM REM Test Larger HMLC models with Noisy Student
FOR /L %%i IN (1, 1, 5) DO ^
python -m src.training.experiment_larger_models ^
-p .\data\fromraw_cid_inchi_smiles_fp_labels.csv ^
-e 1000 ^
-b 256 ^
-l 0.0001 ^
--es-patience 10 ^
--unlabeled-weight 0.5 ^
--log-path .\logs\%atempt%\lager_NoisyStudent\test_%%i
