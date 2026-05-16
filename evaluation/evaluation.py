import torch
import numpy as np
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, f1_score
from models.gnn_model import GNN
from features.smiles_to_graph import smiles_to_graph

# =========================
# LOAD MODEL
# =========================
model = GNN(3, 64, 12)
model.load_state_dict(torch.load("saved_models/gnn.pt", map_location="cpu"))
model.eval()

# =========================
# TARGETS
# =========================
targets = [
    "NR-AR","NR-AR-LBD","NR-AhR","NR-Aromatase",
    "NR-ER","NR-ER-LBD","NR-PPAR-gamma",
    "SR-ARE","SR-ATAD5","SR-HSE","SR-MMP","SR-p53"
]

# =========================
# TEST DATA (example)
# IMPORTANT: replace with real test set
# =========================
test_smiles = [
    "c1ccc(cc1)N",
    "CCO",
    "CC(=O)Cl",
]

test_labels = [
    [0]*12,
    [0]*12,
    [1]*12
]

# =========================
# STORAGE
# =========================
all_preds = []
all_labels = []

# =========================
# PREDICTION LOOP
# =========================
with torch.no_grad():

    for i, smi in enumerate(test_smiles):

        x, edge_index = smiles_to_graph(smi)

        if x is None:
            continue

        output = model(x, edge_index)
        probs = torch.sigmoid(output).numpy()

        all_preds.append(probs)
        all_labels.append(test_labels[i])

# =========================
# CONVERT
# =========================
y_true = np.array(all_labels)
y_pred = np.array(all_preds)

# =========================
# METRICS
# =========================
print("\n🧬 MODEL EVALUATION REPORT\n")

# ROC AUC
try:
    roc = roc_auc_score(y_true, y_pred, average="macro")
    print(f"📊 ROC-AUC: {roc:.4f}")
except:
    print("⚠️ ROC-AUC not enough data")

# BINARY conversion
y_pred_bin = (y_pred > 0.5).astype(int)

# Accuracy
acc = accuracy_score(y_true.flatten(), y_pred_bin.flatten())
print(f"🎯 Accuracy: {acc:.4f}")

# Precision / Recall / F1
precision = precision_score(y_true.flatten(), y_pred_bin.flatten(), zero_division=0)
recall = recall_score(y_true.flatten(), y_pred_bin.flatten(), zero_division=0)
f1 = f1_score(y_true.flatten(), y_pred_bin.flatten(), zero_division=0)

print(f"📌 Precision: {precision:.4f}")
print(f"📌 Recall: {recall:.4f}")
print(f"📌 F1-score: {f1:.4f}")

print("\n✅ Evaluation completed")