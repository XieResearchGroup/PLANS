CUDA_VISIBLE_DEVICES=$1 python -m src.training.experiment_linear_model_mixup \
-p ./data/fromraw_cid_inchi_smiles_fp_labels_onehots.csv \
-e 500 \
-b 256 \
--es-patience 20 \
--log-path ./logs/linear \
--repeat 5 \
--mixup 0.4 \
--mixup-repeat 10
