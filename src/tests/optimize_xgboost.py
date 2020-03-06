from itertools import product

from ..training.experiment_conventional_multiclass import experiment_xgboost

max_depth = [14]
n_round = [17]
rand_seeds = [0, 1, 2, 3, 4, 5]

best = dict()
for m, n, r in product(max_depth, n_round, rand_seeds):
    best.setdefault(r, [0, 0, 0])
    acc, _ = experiment_xgboost(
        data_path=r"D:\Documents\repos\STDSED\data\fromraw_cid_inchi_smiles"
                  "_fp_labels_onehots_twoclass.csv",
        log_path=r"D:\Documents\repos\STDSED\logs\convention\xgboost",
        max_depth=m,
        n_round=n,
        rand_seed=r
    )
    print("max depth: {}, rounds: {}, acc: {}".format(m, n, acc))
    if acc > best.get(r)[2]:
        best.get(r)[0] = m
        best.get(r)[1] = n
        best.get(r)[2] = acc
print("="*80)
for k, v in best.items():
    print(
        "Random seed: {}\n Best params:\nmax_depth - {}\nn_rounds - {}\nacc "
        "- {}\n".format(str(k), v[0], v[1], v[2])
    )
    print("="*80)
