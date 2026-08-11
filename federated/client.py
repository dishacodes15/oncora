"""
client.py
Flower federated learning client.
Each simulated hospital runs an instance of this with its own local data.
The client trains locally and sends only model weights to the server -
raw data never leaves the hospital. This is the core privacy guarantee
of federated learning.

FedProx note: this client implements the proximal-term regularization
that makes FedProx different from FedAvg - see fit() below. The server
side (server.py) only needs to swap FedAvg for FedProx and send mu;
the actual local-training-objective change lives entirely here.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models, transforms, datasets
from torch.utils.data import DataLoader

import flwr as fl
import numpy as np
import sys

# ---- Config ----
DEVICE = torch.device("cpu")
IMG_SIZE = 224
BATCH_SIZE = 32
LOCAL_EPOCHS = 2       # how many epochs each hospital trains before sending weights
LEARNING_RATE = 0.0005
NUM_CLASSES = 4

MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]

train_transforms = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomHorizontalFlip(p=0.3),
    transforms.RandomRotation(10),
    transforms.ToTensor(),
    transforms.Normalize(mean=MEAN, std=STD),
])

eval_transforms = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=MEAN, std=STD),
])


def build_model():
    """Same architecture as train.py - must match exactly for weight aggregation to work."""
    model = models.resnet18(weights=None)
    for param in model.parameters():
        param.requires_grad = False
    for param in model.layer4.parameters():
        param.requires_grad = True
    num_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(0.4),
        nn.Linear(num_features, NUM_CLASSES)
    )
    return model.to(DEVICE)


def get_weights(model):
    """Extract model weights as a list of numpy arrays - what Flower sends to the server."""
    return [val.cpu().numpy() for _, val in model.state_dict().items()]


def set_weights(model, weights):
    """Load weights received from the server back into the model."""
    params_dict = zip(model.state_dict().keys(), weights)
    state_dict = {k: torch.tensor(v) for k, v in params_dict}
    model.load_state_dict(state_dict, strict=True)


class HospitalClient(fl.client.NumPyClient):
    def __init__(self, hospital_id: int):
        self.hospital_id = hospital_id
        self.model = build_model()
        self.criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

        data_dir = f"../datasets/federated/hospital_{hospital_id}"
        train_data = datasets.ImageFolder(root=data_dir, transform=train_transforms)
        val_data = datasets.ImageFolder(root=data_dir, transform=eval_transforms)

        self.train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
        self.val_loader = DataLoader(val_data, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

        print(f"Hospital {hospital_id} initialized with {len(train_data)} images")

    def get_parameters(self, config):
        """Called by server to get this client's current weights."""
        return get_weights(self.model)

    def fit(self, parameters, config):
        """
        Server sends global weights -> client loads them -> trains locally ->
        sends updated weights back. Raw data stays local the whole time.

        FedProx addition: after loading the global weights but before
        local training starts, a detached copy of the (trainable) global
        parameters is kept as global_params. During each training step,
        a proximal term (mu/2) * sum(||w_local - w_global||_2) is added
        to the loss - this penalizes the local model for drifting too
        far from the global model during local training, which is the
        entire algorithmic difference between FedProx and FedAvg. The
        aggregation on the server side (weighted averaging of returned
        weights) is unchanged from FedAvg - only this local objective
        differs. mu is read from config (sent automatically by the
        server's FedProx strategy - see server.py), defaulting to 0.0
        (equivalent to plain FedAvg) if not present, so this client
        still works correctly against a plain FedAvg server too.
        """
        set_weights(self.model, parameters)

        trainable_params = [p for p in self.model.parameters() if p.requires_grad]
        global_params = [p.clone().detach() for p in trainable_params]
        proximal_mu = config.get("proximal_mu", 0.0)

        optimizer = optim.Adam(trainable_params, lr=LEARNING_RATE, weight_decay=1e-4)

        self.model.train()
        for epoch in range(LOCAL_EPOCHS):
            total_loss, correct, total = 0.0, 0, 0
            for images, labels in self.train_loader:
                images, labels = images.to(DEVICE), labels.to(DEVICE)
                optimizer.zero_grad()
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)

                if proximal_mu > 0:
                    proximal_term = 0.0
                    for local_w, global_w in zip(trainable_params, global_params):
                        proximal_term += (local_w - global_w).norm(2)
                    loss = loss + (proximal_mu / 2) * proximal_term

                loss.backward()
                optimizer.step()

                total_loss += loss.item() * images.size(0)
                _, predicted = torch.max(outputs, 1)
                correct += (predicted == labels).sum().item()
                total += labels.size(0)

            print(f"  Hospital {self.hospital_id} | Local Epoch {epoch+1}/{LOCAL_EPOCHS} "
                  f"| Loss: {total_loss/total:.4f} | Acc: {correct/total:.4f} "
                  f"| mu={proximal_mu}")

        return get_weights(self.model), len(self.train_loader.dataset), {}

    def evaluate(self, parameters, config):
        """Server asks client to evaluate the global model on local data."""
        set_weights(self.model, parameters)
        self.model.eval()

        total_loss, correct, total = 0.0, 0, 0
        with torch.no_grad():
            for images, labels in self.val_loader:
                images, labels = images.to(DEVICE), labels.to(DEVICE)
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                total_loss += loss.item() * images.size(0)
                _, predicted = torch.max(outputs, 1)
                correct += (predicted == labels).sum().item()
                total += labels.size(0)

        accuracy = correct / total
        print(f"  Hospital {self.hospital_id} | Eval Accuracy: {accuracy:.4f}")
        return total_loss / total, len(self.val_loader.dataset), {"accuracy": accuracy}


if __name__ == "__main__":
    hospital_id = int(sys.argv[1]) if len(sys.argv) > 1 else 1
    fl.client.start_numpy_client(
        server_address="127.0.0.1:8080",
        client=HospitalClient(hospital_id=hospital_id),
    )



