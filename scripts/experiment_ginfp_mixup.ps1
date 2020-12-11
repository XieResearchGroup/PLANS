python -m src.training.experiment_ginfp_mixup_exploit_partial `
    -p .\data\fromraw_cid_inchi_smiles_fp_labels_onehots.csv `
    -e 2 `
    -b 128 `
    --es-patience 10 `
    --log-path .\logs\linear\ginfp_mixup_partial `
    --repeat 3 `
    --mixup 0.4 `
    --mixup-repeat 10 `
    --rand-seed 1234768