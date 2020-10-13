python -m src.training.experiment_ginfp_mixup `
    -p .\data\cyp450_smiles_GINfp_labels.json `
    -e 200 `
    -b 256 `
    -l 0.0005 `
    --es-patience 100 `
    --unlabeled-weight 1 `
    --log-path .\logs\ginfp_mixup