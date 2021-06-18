# Calculate ECFPs for MolNet data sets
import pandas as pd
import numpy as np

from src.utils.label_convertors import smiles2ecfp


def _add_ecfp_and_label(df, tasks, smiles_col):
    na2us = {1.0: "1", 0.0: "0", np.nan: "_"}
    df["Label"] = ""
    for task in tasks:
        df[task] = df[task].map(na2us)
        df["Label"] = df["Label"] + df[task]
    df["ECFP"] = df[smiles_col].map(smiles2ecfp)
    

def process_tox21(path, outpath):
    df = pd.read_csv(path)
    tasks = [
        "NR-AR",
        "NR-AR-LBD",
        "NR-AhR",
        "NR-Aromatase",
        "NR-ER",
        "NR-ER-LBD",
        "NR-PPAR-gamma",
        "SR-ARE",
        "SR-ATAD5",
        "SR-HSE",
        "SR-MMP",
        "SR-p53",
    ]
    _add_ecfp_and_label(df, tasks, "smiles")
    df.loc[pd.notna(df["ECFP"])].to_csv(outpath)


def process_toxcast(path, outpath):
    df = pd.read_csv(path)
    tasks = list(df.columns)[1:]
    _add_ecfp_and_label(df, tasks, "smiles")
    df.loc[pd.notna(df["ECFP"])].to_csv(outpath)


def process_muv(path, outpath):
    df = pd.read_csv(path)
    tasks = [
        "MUV-466",
        "MUV-548",
        "MUV-600",
        "MUV-644",
        "MUV-652",
        "MUV-689",
        "MUV-692",
        "MUV-712",
        "MUV-713",
        "MUV-733",
        "MUV-737",
        "MUV-810",
        "MUV-832",
        "MUV-846",
        "MUV-852",
        "MUV-858",
        "MUV-859",
    ]
    _add_ecfp_and_label(df, tasks, "smiles")
    df.loc[pd.notna(df["ECFP"])].to_csv(outpath)


if __name__ == "__main__":
    process_tox21(
        "data/MolNet/tox21.csv",
        "data/MolNet/tox21_ecfp.csv"
    )
    process_toxcast(
        "data/MolNet/toxcast_data.csv",
        "data/MolNet/toxcast_ecfp.csv"
    )
    process_muv(
        "data/MolNet/muv.csv",
        "data/MolNet/muv_ecfp.csv"
    )
