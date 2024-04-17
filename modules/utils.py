import csv
import logging
import os
import numpy as np
import plotly.graph_objects as go

PATH_RESULTS = os.getenv("PATH_RESULTS", "results")
os.makedirs(PATH_RESULTS, exist_ok=True)


class Utils:
    @staticmethod
    def setup_logging() -> None:
        logging.basicConfig(
            level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
        )

    @staticmethod
    def softmax(x: np.ndarray) -> np.ndarray:
        y = np.exp(x - np.max(x))
        f_x = y / np.sum(np.exp(x))
        return f_x

    @staticmethod
    def get_metrics(csv_path: str) -> tuple[list[float], list[float], list[float]]:
        train_loss: list[float] = []
        val_loss: list[float] = []
        val_acc: list[float] = []
        with open(csv_path, "r", encoding="utf-8", newline="") as f_in:
            reader = csv.DictReader(f_in)
            for line in reader:
                if line["train_loss"] != "":
                    train_loss.append(float(line["train_loss"]))
                if line["val_loss"] != "":
                    val_loss.append(float(line["val_loss"]))
                if line["val_acc"] != "":
                    val_acc.append(float(line["val_acc"]))
        return train_loss, val_loss, val_acc

    @staticmethod
    def plot_losses(train_loss: list[float], val_loss: list[float]) -> None:
        fig = go.Figure()

        # Training losses
        fig.add_trace(
            go.Scatter(
                x=list(range(1, len(train_loss) + 1)),
                y=train_loss,
                mode="lines+markers",
                name="Training",
            )
        )

        # Validation losses
        fig.add_trace(
            go.Scatter(
                x=list(range(1, len(val_loss) + 1)),
                y=val_loss,
                mode="lines+markers",
                name="Validation",
            )
        )

        fig.update_layout(
            title="Loss vs. No. of epochs", xaxis_title="Epoch", yaxis_title="Loss"
        )

        fig.write_image(os.path.join(PATH_RESULTS, "loss_graph.png"))

    @staticmethod
    def plot_scores(val_acc: list[float]) -> None:
        fig = go.Figure()

        # Validation accuracy
        fig.add_trace(
            go.Scatter(
                x=list(range(1, len(val_acc) + 1)),
                y=val_acc,
                mode="lines+markers",
                name="Validation Accuracy",
            )
        )

        fig.update_layout(
            title="Accuracy vs. No. of epochs",
            xaxis_title="Epoch",
            yaxis_title="Accuracy",
        )

        # fig.show()
        fig.write_image(os.path.join(PATH_RESULTS, "accuracy_graph.png"))


if __name__ == "__main__":
    train_loss, val_loss, val_acc = Utils.get_metrics(
        "lightning_logs/version_0/metrics.csv"
    )

    Utils.plot_losses(train_loss, val_loss)
    Utils.plot_scores(val_acc)
