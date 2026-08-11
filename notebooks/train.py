"""
train.py
Builds and trains a ResNet18-based classifier on the Brain Tumor MRI dataset
using transfer learning. Saves the trained model + prints evaluation metrics.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

from dataset import get_dataloaders

# ---- Config ----
DEVICE = torch.device("cpu")  # MX130 GPU is too old (CUDA capability 5.0) for current PyTorch - using CPU
EPOCHS = 10
LEARNING_RATE = 0.0005  # lowered from 0.001 since layer4 now trains too - keeps its pretrained weights from shifting too aggressively
MODEL_SAVE_PATH = "../models/mri_classifier.pt"

print(f"Using device: {DEVICE}")


def build_model(num_classes):
    """
    Loads a pretrained ResNet18 and replaces its final layer.
    This is transfer learning: we keep all the learned image features
    from ImageNet, and only retrain the last layer for our 4 classes.
    """
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)

    # Freeze all earlier layers - they already know how to detect edges,
    # textures, shapes. We don't want to destroy that knowledge.
    for param in model.parameters():
        param.requires_grad = False

    # Unfreeze layer4 (the last residual block). Early layers detect generic
    # features (edges, textures) that transfer fine from ImageNet as-is, but
    # late layers detect more abstract, task-specific patterns. Letting layer4
    # retrain lets the model learn MRI-specific distinctions (e.g. the subtle
    # differences between glioma/meningioma/notumor) instead of relying on
    # generic ImageNet features for everything except the final classification.
    for param in model.layer4.parameters():
        param.requires_grad = True

    # Replace the final classification layer to output 4 classes instead of
    # 1000. Dropout added before it: with only layer4+fc trainable and no
    # regularization at all, the model was overfitting - confidently wrong
    # on real-world input despite near-100% confidence on the Testing set.
    # Dropout has no effect during model.eval() (inference), so this only
    # changes training behavior, not runtime prediction behavior.
    num_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(0.4),
        nn.Linear(num_features, num_classes)
    )

    return model.to(DEVICE)


def train_one_epoch(model, loader, criterion, optimizer):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in loader:
        images, labels = images.to(DEVICE), labels.to(DEVICE)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        _, predicted = torch.max(outputs, 1)
        correct += (predicted == labels).sum().item()
        total += labels.size(0)

    return running_loss / total, correct / total


def evaluate(model, loader, criterion):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * images.size(0)
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    return running_loss / total, correct / total, all_preds, all_labels


def main():
    train_loader, val_loader, test_loader, class_names = get_dataloaders()
    num_classes = len(class_names)

    model = build_model(num_classes)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)  # caps overconfidence
    # Now training both the new final layer AND layer4. layer4 already has
    # decent pretrained weights, so we don't want to blow them away with a
    # big learning rate - Adam will adjust fc faster than layer4 in practice,
    # but a single conservative LR for both keeps this simple and stable.
    # weight_decay adds L2 regularization - the model previously had none.
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.Adam(trainable_params, lr=LEARNING_RATE, weight_decay=1e-4)

    history = {"train_acc": [], "val_acc": [], "train_loss": [], "val_loss": []}
    best_val_acc = 0.0  # tracks the best validation accuracy seen so far across epochs

    print("\nStarting training...\n")
    for epoch in range(EPOCHS):
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer)
        val_loss, val_acc, _, _ = evaluate(model, val_loader, criterion)

        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)

        print(f"Epoch {epoch+1}/{EPOCHS} | "
              f"Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} | "
              f"Val Loss: {val_loss:.4f} Acc: {val_acc:.4f}")

        # Save a checkpoint only when validation accuracy improves. This way,
        # even if later epochs overfit and val accuracy drops, we still keep
        # the best version the model reached - not just whatever the last
        # epoch happened to be.
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                "model_state_dict": model.state_dict(),
                "class_names": class_names,
                "val_accuracy": val_acc,
                "epoch": epoch + 1,
            }, MODEL_SAVE_PATH)
            print(f"  -> New best val accuracy ({val_acc:.4f}). Checkpoint saved.")

    # ---- Load best checkpoint for final test evaluation ----
    # The loop above already saved the best-performing version to disk as it
    # trained. We reload it here so the test evaluation (and saved plots)
    # reflect the best model, not whatever the final epoch happened to produce.
    best_checkpoint = torch.load(MODEL_SAVE_PATH, map_location=DEVICE, weights_only=False)
    model.load_state_dict(best_checkpoint["model_state_dict"])
    print(f"\nLoaded best checkpoint from epoch {best_checkpoint['epoch']} "
          f"(val accuracy: {best_checkpoint['val_accuracy']:.4f})")

    # ---- Final test evaluation ----
    print("\nEvaluating on held-out test set...\n")
    test_loss, test_acc, preds, labels = evaluate(model, test_loader, criterion)
    print(f"Test Accuracy: {test_acc:.4f}")
    print("\nClassification Report:")
    print(classification_report(labels, preds, target_names=class_names))

    print(f"\nBest model already saved to {MODEL_SAVE_PATH} (during training loop)")

    # ---- Plot accuracy curve ----
    plt.figure(figsize=(8, 5))
    plt.plot(history["train_acc"], label="Train Accuracy")
    plt.plot(history["val_acc"], label="Validation Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Training vs Validation Accuracy")
    plt.legend()
    plt.savefig("../docs/accuracy_curve.png")
    print("Accuracy curve saved to docs/accuracy_curve.png")

    # ---- Plot loss curve ----
    # Loss reveals overfitting earlier/more clearly than accuracy - val
    # loss can visibly start rising while val accuracy still looks stable.
    plt.figure(figsize=(8, 5))
    plt.plot(history["train_loss"], label="Train Loss")
    plt.plot(history["val_loss"], label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training vs Validation Loss")
    plt.legend()
    plt.savefig("../docs/loss_curve.png")
    print("Loss curve saved to docs/loss_curve.png")

    # ---- Plot confusion matrix ----
    cm = confusion_matrix(labels, preds)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", xticklabels=class_names, yticklabels=class_names, cmap="Blues")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")
    plt.savefig("../docs/confusion_matrix.png")
    print("Confusion matrix saved to docs/confusion_matrix.png")


if __name__ == "__main__":
    main()
