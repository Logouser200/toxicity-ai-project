from rdkit import Chem

# Example molecule
smiles = "CCO"

# Convert SMILES to molecule
mol = Chem.MolFromSmiles(smiles)

# Print number of atoms
print("Number of atoms:", mol.GetNumAtoms())

# Print atoms
print("\nAtoms:")

for atom in mol.GetAtoms():
    print(atom.GetSymbol())