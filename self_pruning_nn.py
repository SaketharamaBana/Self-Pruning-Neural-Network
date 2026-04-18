# ============================================
# SELF-PRUNING NEURAL NETWORK (FINAL VERSION)
# ============================================

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

# ============================================
# DEVICE SETUP
# ============================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# ============================================
# PRUNABLE LINEAR LAYER
# ============================================
class PrunableLinear(nn.Module):
    def __init__(self, in_features, out_features):
        super(PrunableLinear, self).__init__()

        self.weight = nn.Parameter(torch.randn(in_features, out_features) * 0.01)
        self.bias = nn.Parameter(torch.zeros(out_features))

        # Learnable gates
        self.gate_scores = nn.Parameter(torch.randn(in_features, out_features))

    def forward(self, x):
        gates = torch.sigmoid(self.gate_scores)   # 0 → 1
        pruned_weights = self.weight * gates
        return torch.matmul(x, pruned_weights) + self.bias


# ============================================
# MODEL
# ============================================
class PrunableNet(nn.Module):
    def __init__(self):
        super(PrunableNet, self).__init__()

        self.fc1 = PrunableLinear(32*32*3, 256)
        self.fc2 = PrunableLinear(256, 128)
        self.fc3 = PrunableLinear(128, 10)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x


# ============================================
# DATA LOADING
# ============================================
transform = transforms.Compose([
    transforms.ToTensor()
])

trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)

testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=64, shuffle=False)


# ============================================
# SPARSITY LOSS (L1)
# ============================================
def sparsity_loss(model):
    loss = 0
    for module in model.modules():
        if isinstance(module, PrunableLinear):
            gates = torch.sigmoid(module.gate_scores)
            loss += torch.sum(gates)
    return loss


# ============================================
# TRAINING FUNCTION
# ============================================
def train_model(lambda_val=0.01, epochs=10):
    model = PrunableNet().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    print(f"\nTraining with lambda = {lambda_val}")

    for epoch in range(epochs):
        model.train()
        total_loss = 0

        for images, labels in trainloader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()

            outputs = model(images)

            loss_cls = criterion(outputs, labels)
            loss_sp = sparsity_loss(model)

            loss = loss_cls + lambda_val * loss_sp

            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch [{epoch+1}/{epochs}] Loss: {total_loss:.4f}")

    return model


# ============================================
# EVALUATION
# ============================================
def evaluate_model(model):
    model.eval()

    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in testloader:
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            _, predicted = torch.max(outputs, 1)

            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    return 100 * correct / total


# ============================================
# HARD PRUNING (IMPORTANT FIX)
# ============================================
def apply_pruning(model, threshold=1e-2):
    for module in model.modules():
        if isinstance(module, PrunableLinear):
            with torch.no_grad():
                gates = torch.sigmoid(module.gate_scores)
                mask = (gates >= threshold).float()
                module.weight.data *= mask


# ============================================
# SPARSITY CALCULATION
# ============================================
def calculate_sparsity(model, threshold=1e-2):
    total = 0
    pruned = 0

    for module in model.modules():
        if isinstance(module, PrunableLinear):
            gates = torch.sigmoid(module.gate_scores)

            total += gates.numel()
            pruned += torch.sum(gates < threshold).item()

    return 100 * pruned / total


# ============================================
# PLOT DISTRIBUTION
# ============================================
def plot_gate_distribution(model):
    all_gates = []

    for module in model.modules():
        if isinstance(module, PrunableLinear):
            gates = torch.sigmoid(module.gate_scores).detach().cpu().numpy()
            all_gates.extend(gates.flatten())

    plt.hist(all_gates, bins=50)
    plt.title("Gate Value Distribution")
    plt.xlabel("Gate Value")
    plt.ylabel("Frequency")
    plt.show()


# ============================================
# MAIN EXPERIMENT
# ============================================
if __name__ == "__main__":

    lambda_values = [0.001, 0.01, 0.1]
    results = []

    best_model = None
    best_acc = 0

    for lam in lambda_values:
        model = train_model(lambda_val=lam, epochs=10)

        # Apply hard pruning
        apply_pruning(model)

        acc = evaluate_model(model)
        sparsity = calculate_sparsity(model)

        results.append((lam, acc, sparsity))

        print(f"\nLambda: {lam}")
        print(f"Accuracy: {acc:.2f}%")
        print(f"Sparsity: {sparsity:.2f}%")

        # Track best model
        if acc > best_acc:
            best_acc = acc
            best_model = model

    # ============================================
    # FINAL RESULTS TABLE
    # ============================================
    print("\nFINAL RESULTS:")
    print("Lambda\tAccuracy\tSparsity")
    for r in results:
        print(f"{r[0]}\t{r[1]:.2f}%\t\t{r[2]:.2f}%")

    # ============================================
    # PLOT BEST MODEL
    # ============================================
    print("\nPlotting best model gate distribution...")
    plot_gate_distribution(best_model)