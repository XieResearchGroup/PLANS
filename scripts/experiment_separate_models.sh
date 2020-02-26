CUDA_VISIBLE_DEVICES=$1 \
python -m src.training.experiment_separate_models \
-p ./data/fromraw_cid_inchi_smiles_fp_labels_onehots.csv \
-e 200 \
-b 128 \
--es-patience 10 \
--log-path ./logs/linear_separate \
--repeat 3 \
--mixup 0.4 \
--mixup-repeat 10