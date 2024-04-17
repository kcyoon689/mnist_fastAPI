import torch
import torch.nn as nn
import torch.nn.functional as F


class MnistModelBase(nn.Module):
    def training_step(self, batch) -> torch.Tensor:
        images, targets = batch
        out = self(images)
        loss = F.cross_entropy(out, targets)
        return loss

    def accuracy(self, outputs: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        _, preds = torch.max(outputs, dim=1)
        return torch.tensor(torch.sum(preds == labels).item() / len(preds))

    def validation_step(self, batch) -> dict[str, torch.Tensor]:
        images, targets = batch
        out = self(images)  # Generate predictions
        loss = F.cross_entropy(out, targets)  # Calculate loss
        acc = self.accuracy(out, targets)
        return {"val_loss": loss.detach(), "val_acc": acc}

    def validation_epoch_end(
        self, outputs: list[dict[str, torch.Tensor]]
    ) -> dict[str, float]:
        batch_losses = [x["val_loss"] for x in outputs]
        epoch_loss = torch.stack(batch_losses).mean()  # Combine losses
        batch_acc = [x["val_acc"] for x in outputs]
        epoch_acc = torch.stack(batch_acc).mean()  # Combine accuracies
        return {"val_loss": epoch_loss.item(), "val_acc": epoch_acc.item()}

    def epoch_end(self, epoch, result, LR) -> None:
        print(
            f"Epoch [{epoch}] - LR [{LR}], train_loss: {result['train_loss']:.4f}, "
            + f"val_loss: {result['val_loss']:.4f}, val_acc: {result['val_acc']:.4f}"
        )
