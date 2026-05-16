import torch
from models.gnn_model import GNN
from utils.data_loader import load_data

# ======================
# LOAD DATA
# ======================
graphs, labels = load_data("data/train.csv")

# ======================
# MODEL
# ======================
input_dim = 3
hidden_dim = 64
output_dim = 12

model = GNN(input_dim, hidden_dim, output_dim)

optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

pos_weight = torch.tensor([3.0] * 12)
criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)

# ======================
# TRAINING
# ======================
epochs = 50

for epoch in range(epochs):

    total_loss = 0

    for i in range(len(graphs)):

        x, edge_index = graphs[i]
        y = labels[i]

        # FIX label
        y = torch.tensor(y, dtype=torch.float)

        optimizer.zero_grad()

        out = model(x, edge_index)

        loss = criterion(out, y)

        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(f"Epoch {epoch+1}, Loss: {total_loss:.4f}")

# ======================
# SAVE MODEL
# ======================
torch.save(model.state_dict(), "saved_models/gnn.pt")

print("✅ GNN training completed")