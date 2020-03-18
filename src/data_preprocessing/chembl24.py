import pandas as pd
import h5py
import numpy as np
from rdkit.Chem import MolFromSmiles
from rdkit.Chem.rdMolDescriptors import GetMorganFingerprintAsBitVect


def smiles2ecfp(smiles, radius=4, bits=2048):
    mol = MolFromSmiles(smiles)
    if mol is None:
        return ""
    fp = GetMorganFingerprintAsBitVect(mol, radius, nBits=bits)
    return "".join(map(str, list(fp)))


if __name__ == "__main__":
    path = "data/ChEMBL24_all_compounds.csv.gz"
    data = pd.read_csv(path)
    # Calculate ECFP
    extracted_data = data[["ChEMBL_ID", " SMILES"]]
    extracted_data["ECFP"] = extracted_data[" SMILES"].map(smiles2ecfp)
    extracted_data.drop(
        extracted_data[extracted_data["ECFP"] == ""].index, inplace=True)
    extracted_data.to_csv("data/ChEMBL24_smiles_fp.csv")
    # Convert the csv file to hdf5 file
    chembl = pd.read_csv("data/ChEMBL24_smiles_fp.csv")
    h5f = h5py.File("../data/ChEMBL24.hdf5", "w")
    root_gp = h5f.create_group("/ChEMBL")
    dt = h5py.string_dtype(encoding="utf-8")
    root_gp.create_dataset("ChEMBL_ID",
                           data=chembl["ChEMBL_ID"].astype(bytes),
                           dtype=dt)
    root_gp.create_dataset("SMILES",
                           data=chembl[" SMILES"].astype(bytes),
                           dtype=dt)
    np_ecfp = chembl["ECFP"].map(lambda x: np.fromiter(x, dtype=int))
    root_gp.create_dataset("ECFP", data=np.stack(list(np_ecfp), axis=0))
    h5f.close()
