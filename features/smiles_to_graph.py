from rdkit import Chem
import torch


def smiles_to_graph(smiles):

    mol = Chem.MolFromSmiles(smiles)

    if mol is None:
        return None, None

    if mol.GetNumAtoms() == 0:
        return None, None

    if mol.GetNumBonds() == 0:
        return None, None

    nodes = []
    edges = []

    for atom in mol.GetAtoms():

      features = [
        atom.GetAtomicNum(),     # type ديال atom (C, O, N)
        atom.GetDegree(),        # عدد الروابط
        atom.GetIsAromatic()     # واش aromatic ولا لا
    ]

      nodes.append(features)

    for bond in mol.GetBonds():
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()

        edges.append((i, j))
        edges.append((j, i))

    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()

    return torch.tensor(nodes, dtype=torch.float32), edge_index