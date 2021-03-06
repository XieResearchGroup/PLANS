CUDA_VISIBLE_DEVICES=$1 \
python -m src.training.experiment_linear_drugbank_mixup \
-p ./data/fromraw_cid_inchi_smiles_fp_labels_onehots.csv \
--outside-path ./data/DrugBank_smiles_fp.csv \
-e 1000 \
-b 256 \
--es-patience 20 \
--log-path ./logs/linear/drugbank_mixup \
--repeat 5 \
--mixup 0.4 \
--mixup-repeat 10
