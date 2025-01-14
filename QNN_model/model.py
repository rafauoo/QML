import torch
import torch.nn as nn
from qiskit.circuit.library import ZZFeatureMap
from qiskit_machine_learning.neural_networks import SamplerQNN
from qiskit_machine_learning.connectors import TorchConnector
from qiskit.primitives import Sampler
import pytorch_lightning as pl
import csv
from torchmetrics.functional import accuracy
from pathlib import Path


def create_qnn(num_features):
    """Creates a Quantum Neural Network (QNN) using Qiskit's ZZFeatureMap."""
    feature_map = ZZFeatureMap(num_features)
    sampler = Sampler()
    qnn = SamplerQNN(
        circuit=feature_map,
        input_params=feature_map.parameters,
        sampler=sampler
    )
    return qnn

class QuantumSmileClassifier(pl.LightningModule):
    def __init__(self, num_features, learning_rate=0.0005):
        """Initialize the Quantum Smile Classifier model."""
        super().__init__()
        self.csv_path = Path("./QNN_model/epoch_metrics.csv")
        qnn = create_qnn(num_features)
        self._initialize_csv()
        self.quantum_layer = TorchConnector(qnn)
        self.fc_layer = nn.Linear(1024, 2)
        self.loss_func = nn.CrossEntropyLoss()
        self.learning_rate = learning_rate

    def forward(self, x):
        """Forward pass through the model (quantum + fully connected layers)."""
        q_output = self.quantum_layer(x)
        output = self.fc_layer(q_output)
        return output

    def training_step(self, batch, batch_idx):
        """Training step: Calculate loss and accuracy for a batch."""
        faces_features = batch["faces_features"]
        authenticities = batch["authenticity"]
        outputs = self(faces_features)
        loss = self.loss_func(outputs, authenticities)
        predictions = torch.argmax(outputs, dim=1)
        acc = accuracy(predictions, authenticities, task="binary")

        self.log("train_loss", loss, prog_bar=True, logger=True)
        self.log("train_accuracy", acc, prog_bar=True, logger=True)

        return {"loss": loss, "accuracy": acc}

    def validation_step(self, batch, batch_idx):
        """Validation step: Calculate loss and accuracy for a batch."""
        faces_features = batch["faces_features"]
        authenticities = batch["authenticity"]
        outputs = self(faces_features)

        loss = self.loss_func(outputs, authenticities)
        predictions = torch.argmax(outputs, dim=1)
        acc = accuracy(predictions, authenticities, task="binary")

        self.log("val_loss", loss, prog_bar=True, logger=True)
        self.log("val_accuracy", acc, prog_bar=True, logger=True)

        return {"loss": loss, "accuracy": acc}

    def test_step(self, batch, batch_idx):
        """Test step: Calculate loss and accuracy for a batch."""
        faces_features = batch["faces_features"]
        authenticities = batch["authenticity"]
        outputs = self(faces_features)
        loss = self.loss_func(outputs, authenticities)
        predictions = torch.argmax(outputs, dim=1)
        acc = accuracy(predictions, authenticities, task="binary")

        self.log("test_loss", loss, prog_bar=True, logger=True)
        self.log("test_accuracy", acc, prog_bar=True, logger=True)

        return {"loss": loss, "accuracy": acc}

    def configure_optimizers(self):
        """Configure the optimizer and learning rate scheduler."""
        optimizer = torch.optim.Adam(
            self.parameters(),
            lr=self.learning_rate,
            weight_decay=1e-5
        )

        scheduler = {
            "scheduler": torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode="min",
                factor=0.97,
                patience=25,
                verbose=True
            ),
            "monitor": "val_loss",
            "interval": "epoch",
            "frequency": 1
        }

        return {"optimizer": optimizer, "lr_scheduler": scheduler}

    def on_train_epoch_end(self):
        self._log_epoch_metrics("train")

    def on_validation_epoch_end(self):
        self._log_epoch_metrics("val")

    def on_test_epoch_end(self):
        self._log_epoch_metrics("test")

    def _initialize_csv(self):
        if not self.csv_path.exists():
            with open(self.csv_path, mode="w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["epoch", "phase", "loss", "accuracy"])

    def _log_epoch_metrics(self, phase):
        metrics = self.trainer.callback_metrics
        loss = metrics.get(f"{phase}_loss")
        accuracy = metrics.get(f"{phase}_accuracy")
        with open(self.csv_path, mode="a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([self.current_epoch, phase, loss.item() if loss else None,
                             accuracy.item() if accuracy else None])