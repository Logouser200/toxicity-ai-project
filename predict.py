import torch
from models.gnn_model import GNN
from features.smiles_to_graph import smiles_to_graph

# =========================
# LOAD MODEL
# =========================
model = GNN(3, 64, 12)
model.load_state_dict(torch.load("saved_models/gnn.pt", map_location=torch.device('cpu')))
model.eval()

# =========================
# INPUT MOLECULE
# =========================
smiles = "c1ccc(cc1)N"

x, edge_index = smiles_to_graph(smiles)

if x is None:
    print("❌ Invalid SMILES")
    exit()

# =========================
# PREDICTION
# =========================
with torch.no_grad():
    output = model(x, edge_index)
    probs = torch.sigmoid(output)

# =========================
# LABELS
# =========================
targets = [
    "NR-AR","NR-AR-LBD","NR-AhR","NR-Aromatase",
    "NR-ER","NR-ER-LBD","NR-PPAR-gamma",
    "SR-ARE","SR-ATAD5","SR-HSE","SR-MMP","SR-p53"
]

labels_map = {
    "NR-AR": {"fr": "Récepteur des androgènes nucléaire", "ar": "مستقبل الأندروجين النووي"},
    "NR-AR-LBD": {"fr": "Domaine de liaison des androgènes", "ar": "مجال ارتباط الأندروجين"},
    "NR-AhR": {"fr": "Récepteur des hydrocarbures aromatiques", "ar": "مستقبل الهيدروكربونات العطرية"},
    "NR-Aromatase": {"fr": "Aromatase", "ar": "إنزيم الأروماتاز"},
    "NR-ER": {"fr": "Récepteur des œstrogènes", "ar": "مستقبل الإستروجين"},
    "NR-ER-LBD": {"fr": "Domaine de liaison des œstrogènes", "ar": "مجال ارتباط الإستروجين"},
    "NR-PPAR-gamma": {"fr": "PPAR-gamma (régulation métabolique)", "ar": "منظم الأيض PPAR-غاما"},
    "SR-ARE": {"fr": "Élément de réponse au stress oxydatif", "ar": "عنصر استجابة الإجهاد التأكسدي"},
    "SR-ATAD5": {"fr": "Réponse aux dommages de l'ADN", "ar": "استجابة تلف الحمض النووي"},
    "SR-HSE": {"fr": "Élément de choc thermique", "ar": "عنصر الصدمة الحرارية"},
    "SR-MMP": {"fr": "Potentiel mitochondrial", "ar": "الجهد الميتوكوندري"},
    "SR-p53": {"fr": "Protéine suppresseur de tumeur p53", "ar": "بروتين كابح الأورام p53"}
}

# =========================
# OUTPUT
# =========================
print("\n" + "="*55)
print("🧬 TOXICITY PREDICTION REPORT")
print("="*55)

for i, t in enumerate(targets):

    p = round(probs[i].item() * 100, 2)

    # risk level
    if p > 75:
        status = "🔴 HIGH TOXIC"
    elif p > 50:
        status = "⚠️ TOXIC"
    else:
        status = "✅ SAFE"

    fr = labels_map[t]["fr"]
    ar = labels_map[t]["ar"]

    print("\n-----------------------------")
    print(f"🇫🇷 {fr}")
    print(f"🇸🇦 {ar}")
    print(f"Status: {status}")
    print(f"Probability: {p}%")

print("\n" + "="*55)