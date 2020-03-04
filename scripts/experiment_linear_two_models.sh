for i in {1..5}
do
    CUDA_VISIBLE_DEVICES=$1 \
    python -m src.training.experiment_linear_two_models \
    -p ./data/fromraw_cid_inchi_smiles_fp_labels_onehots_twoclass.csv \
    -e 1000 \
    -b 256 \
    --es-patience 20 \
    --log-path ./logs/linear/two_models \
    --repeat 5 \
    --rand-seed $i
done
