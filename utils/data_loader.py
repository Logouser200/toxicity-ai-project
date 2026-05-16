import pandas as pd
import torch
from features.smiles_to_graph import smiles_to_graph


def load_data(path):

    df = pd.read_csv(path)

    graphs = []
    labels = []

    targets = [
        "NR-AR", "NR-AR-LBD", "NR-AhR", "NR-Aromatase",
        "NR-ER", "NR-ER-LBD", "NR-PPAR-gamma",
        "SR-ARE", "SR-ATAD5", "SR-HSE", "SR-MMP", "SR-p53"
    ]

    # 🔥 important: clean NaN
    df = df.dropna(subset=["smiles"])

    for _, row in df.iterrows():

        smiles = row["smiles"]

        x, edge_index = smiles_to_graph(smiles)

        if x is None:
            continue

        # 🔥 FIX HERE
        y = row[targets].fillna(0).astype(float).values
        y = torch.tensor(y, dtype=torch.float32)

        graphs.append((x, edge_index))
        labels.append(y)

    return graphs, labels