import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torchmetrics.functional import accuracy
import lightning as L
from lightning.pytorch.callbacks import ModelSummary
from lightning.pytorch.utilities.types import OptimizerLRSchedulerConfig
from modules.data_module import MNISTDataModule


class MNISTModel(L.LightningModule):
    def __init__(
        self, channels, width, height, num_classes, hidden_size=64, learning_rate=0.01
    ):
        super().__init__()
        self.example_input_array = torch.Tensor(64, 1, 28, 28)
        self.save_hyperparameters()

        # We take in input dimensions as parameters and use those to dynamically build model.
        self.channels = channels
        self.width = width
        self.height = height
        self.num_classes = num_classes
        self.hidden_size = hidden_size
        self.learning_rate = learning_rate

        self.model = nn.Sequential(
            nn.Flatten(),
            nn.Linear(channels * width * height, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.model(x)
        return F.log_softmax(x, dim=1)

    def training_step(self, batch: torch.Tensor, batch_idx: int) -> torch.Tensor:
        x, y = batch
        logits = self(x)
        loss = F.nll_loss(logits, y)
        self.log("train_loss", loss, prog_bar=True, on_step=False, on_epoch=True)
        return loss

    def validation_step(self, batch: torch.Tensor, batch_idx: int) -> None:
        x, y = batch
        logits = self(x)
        loss = F.nll_loss(logits, y)
        preds = torch.argmax(logits, dim=1)
        acc = accuracy(preds, y, task="multiclass", num_classes=10)
        self.log("val_loss", loss, prog_bar=True)
        self.log("val_acc", acc, prog_bar=True)

    def test_step(self, batch: torch.Tensor, batch_idx: int) -> None:
        return self.validation_step(batch, batch_idx)

    def configure_optimizers(self) -> OptimizerLRSchedulerConfig:
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        lr_scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=self.learning_rate,
            total_steps=int(self.trainer.estimated_stepping_batches),
        )
        return {"optimizer": optimizer, "lr_scheduler": lr_scheduler}


# Print Summary of the model
if __name__ == "__main__":
    dm = MNISTDataModule()
    model = MNISTModel(*dm.dims, num_classes=dm.num_classes)
    trainer = model.trainer

    # Train the model
    trainer.fit(model=model, datamodule=dm)

    # Test the model
    trainer.test(model=model, datamodule=dm)

    # Predict
    print("answer: [2, 0, 9, 0, 3, 7, 0, 3, 0, 3]")
    results = []
    for idx in range(1, 11):
        image = Image.open(f"samples/img_{idx}.jpg")  # 2
        image_tensor = dm.predict_transform(image).unsqueeze(0)
        raw_output = model(image_tensor.to(model.device))
        confidence = F.softmax(raw_output, dim=1)[0] * 100
        result = torch.argmax(raw_output, dim=1)[0]
        # print(raw_output[0].detach().numpy())
        # print(confidence.detach().numpy())
        # print(result.item())
        results.append(result.item())
    print(results)
