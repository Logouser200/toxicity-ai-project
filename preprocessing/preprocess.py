import pandas as pd
from sklearn.model_selection import train_test_split

# =========================
# LOAD DATASET
# =========================

df = pd.read_csv("data/tox21.csv")

print("Dataset Shape:")
print(df.shape)

print("\nFirst Rows:")
print(df.head())

# =========================
# REMOVE MISSING VALUES
# =========================

df = df.dropna()

print("\nDataset After Cleaning:")
print(df.shape)

# =========================
# SELECT IMPORTANT COLUMNS
# =========================

selected_columns = [
    "smiles",
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
    "SR-p53"
]

df = df[selected_columns]

# =========================
# TRAIN / TEST SPLIT
# =========================

train_df, test_df = train_test_split(
    df,
    test_size=0.2,
    random_state=42
)

print("\nTrain Shape:", train_df.shape)
print("Test Shape:", test_df.shape)

# =========================
# SAVE CLEAN DATA
# =========================

train_df.to_csv("data/train.csv", index=False)
test_df.to_csv("data/test.csv", index=False)

print("\n✅ Preprocessing Completed")