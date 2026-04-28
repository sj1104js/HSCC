import torch
import argparse
import torch.nn as nn
import torch.utils.data as Data
from scipy.io import loadmat
import numpy as np
import time
import torch.nn.functional as F
import os
from utils.dataprocess import load_data, normalize, traintwo_patch, apply_pca, split_train_test_labels, padpatch, \
    loadtrandte_data
from model.HSCC import HSCC
from utils.evulate import calculate_metrics, evaluatetwo
from utils.output import save_metrics_and_accuracies

# -------------------------------------------------------------------------------
# Configuration Parameters
parser = argparse.ArgumentParser("HSCC")
parser.add_argument('--seed', type=int, default=0, help='Random seed for reproducibility')
parser.add_argument('--test_freq', type=int, default=10, help='Evaluation frequency in epochs')
parser.add_argument('--epoches', type=int, default=300, help='Total training epochs')
parser.add_argument('--learning_rate', type=float, default=1e-3, choices=[1e-5, 5e-5, 1e-4, 5e-4, 1e-3, 5e-3],
                    help='Learning rate')
parser.add_argument('--gamma', type=float, default=0.9, help='Learning rate decay factor')
parser.add_argument('--weight_decay', type=float, default=0, help='Weight decay for optimizer')
parser.add_argument('--dataset', default='Muufl', choices=['Muufl', 'Trento', 'Houston'], help='Dataset selection')
parser.add_argument('--num_classes', type=int, default=11, choices=[11, 6, 15], help='Number of classes')
parser.add_argument('--batch_size', type=int, default=64, help='Training batch size')
parser.add_argument('--train_num', type=int, default=50, help='Number of training samples per class')
parser.add_argument('--patches1', type=int, default=9, choices=[7, 9, 11, 13, 15, 17, 19],
                    help='Patch size for HSI data')
parser.add_argument("--use_pca", type=bool, default=False)
parser.add_argument("--pca_components", type=int, default=20)
parser.add_argument('--more0', action='store_true', default=False, help='Flag for additional data processing')
parser.add_argument('--lam1', default=2, type=float, help='Balance coefficient for attention loss')
args = parser.parse_args()

# -------------------------------------------------------------------------------
# Helper function to create dataloaders
def create_dataloader(patch_size):
    """Create data loaders for training and testing with specific patch size"""
    # Load dataset based on configuration
    Data1, Data2, gt, train_labels, test_labels = load_data(args.dataset)

    # Convert and normalize data
    Data1 = normalize(Data1.astype(np.float32))
    Data2 = normalize(Data2.astype(np.float32))
    if args.use_pca:
        Data1 = apply_pca(Data1, args.pca_components)

    # Prepare patches for model input
    pad_width = patch_size // 2
    TrainPatch1, TrainPatch2, TrainLabel = traintwo_patch(
        Data1, Data2, patch_size, pad_width, train_labels, args.more0
    )
    TestPatch1, TestPatch2, TestLabel = traintwo_patch(
        Data1, Data2, patch_size, pad_width, test_labels, args.more0
    )

    # Create PyTorch datasets
    train_dataset = Data.TensorDataset(TrainPatch1, TrainPatch2, TrainLabel)
    test_dataset = Data.TensorDataset(TestPatch1, TestPatch2, TestLabel)

    # Create data loaders
    train_loader = Data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True
    )
    test_loader = Data.DataLoader(
        test_dataset, batch_size=args.batch_size, shuffle=False
    )

    return train_loader, test_loader

# -------------------------------------------------------------------------------
# Training function
def train(model, optimizer, criterion, scheduler, train_loader, test_loader, device, patch_size):
    """Model training procedure"""
    best_accuracy = 0.0
    best_model_path = os.path.join(args.dataset, f"best_model_patch_{patch_size}.pth")
    train_losses = []

    for epoch in range(args.epoches):
        model.train()
        training_loss = 0

        # Batch training
        for hsi, lidar, labels in train_loader:
            hsi, lidar, labels = hsi.to(device), lidar.to(device), labels.to(device)

            optimizer.zero_grad()
            attnloss, outputs = model(hsi, lidar)
            loss = criterion(outputs, labels) + args.lam1 * attnloss
            loss.backward()
            optimizer.step()

            training_loss += loss.item()

        # Periodic evaluation
        if (epoch + 1) % args.test_freq == 0:
            accuracy = evaluatetwo(model, test_loader, device)
            avg_loss = training_loss / len(train_loader)
            train_losses.append(avg_loss)

            print(f"Epoch [{epoch + 1}/{args.epoches}], "
                  f"Loss: {avg_loss:.4f}, "
                  f"Test Accuracy: {accuracy:.2f}%")

            # Save best model
            if accuracy > best_accuracy:
                os.makedirs(args.dataset, exist_ok=True)
                best_accuracy = accuracy
                torch.save(model.state_dict(), best_model_path)
                print(f"New best model saved with accuracy: {best_accuracy:.2f}%")

        scheduler.step()

    return train_losses, best_model_path, best_accuracy

# -------------------------------------------------------------------------------
# Testing function
def test(model, test_loader, best_model_path, num_classes, device):
    """Model evaluation procedure"""
    model.load_state_dict(torch.load(best_model_path, map_location=device, weights_only=True))
    model.eval()

    total, correct = 0, 0
    confusion_matrix = np.zeros((num_classes, num_classes))

    with torch.no_grad():
        for hsi, lidar, labels in test_loader:
            hsi, lidar, labels = hsi.to(device), lidar.to(device), labels.to(device)

            _, outputs = model(hsi, lidar)
            _, predicted = torch.max(outputs, 1)

            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            # Update confusion matrix
            for t, p in zip(labels, predicted):
                confusion_matrix[t.item(), p.item()] += 1

    # Calculate metrics
    OA, AA, Kappa, class_accuracy = calculate_metrics(
        confusion_matrix, total, correct
    )

    print(f"\nOverall Accuracy (OA): {OA:.2f}%")
    print(f"Average Accuracy (AA): {AA:.2f}%")
    print(f"Kappa Coefficient: {Kappa:.4f}")

    for i, acc in enumerate([acc * 100 for acc in class_accuracy]):
        print(f"Class {i + 1} Accuracy: {acc:.2f}%")

    return confusion_matrix, class_accuracy, OA, AA, Kappa

# -------------------------------------------------------------------------------
# Focal loss definition (optional, not used in default criterion)
class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss

# -------------------------------------------------------------------------------
# Main execution
if __name__ == "__main__":
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Configure input dimensions based on dataset
    dataset_config = {
        'Houston': (144, 1),
        'Trento': (63, 1),
        'Muufl': (64, 2),
        'Augsburg': (180,1)
    }

    if args.use_pca:
        band1 = args.pca_components
        _, band2 = dataset_config[args.dataset]
    else:
        band1, band2 = dataset_config[args.dataset]

    # Instantiate model
    model = HSCC(band1, band2, args.num_classes).to(device)

    # Create data loaders
    train_loader, test_loader = create_dataloader(args.patches1)

    # Loss, optimizer, scheduler
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=args.gamma)

    # Train
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    start_time = time.time()
    train_losses, best_model_path, _ = train(
        model, optimizer, criterion, scheduler,
        train_loader, test_loader, device, args.patches1
    )
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    train_time = time.time() - start_time

    # Test
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    start_time = time.time()
    confusion_matrix, class_accuracy, OA, AA, Kappa = test(
        model, test_loader, best_model_path, args.num_classes, device
    )
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    test_time = time.time() - start_time

    # Save results
    save_metrics_and_accuracies(
        class_accuracy, OA, AA, Kappa,
        train_time, test_time, args.dataset
    )