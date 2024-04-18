import csv
import datetime
import logging
import os
import numpy as np
import plotly.graph_objects as go


class Utils:
    @staticmethod
    def setup_logging() -> None:
        os.makedirs("logs", exist_ok=True)
        datetime_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        logging.basicConfig(
            level=logging.INFO,
            filename=f"logs/{datetime_str}.log",
            format="%(asctime)s - %(levelname)s - %(message)s",
        )

    @staticmethod
    def softmax(x: np.ndarray) -> np.ndarray:
        y = np.exp(x - np.max(x))
        f_x = y / np.sum(np.exp(x))
        f_x = max(f_x) / sum(f_x) * 100
        return f_x

    @staticmethod
    def get_metrics(csv_path: str) -> tuple[list[float], list[float], list[float]]:
        train_loss_list: list[float] = []
        val_loss_list: list[float] = []
        val_acc_list: list[float] = []
        with open(csv_path, "r", encoding="utf-8", newline="") as f_in:
            reader = csv.DictReader(f_in)
            for line in reader:
                if line["train_loss"] != "":
                    train_loss_list.append(float(line["train_loss"]))
                if line["val_loss"] != "":
                    val_loss_list.append(float(line["val_loss"]))
                if line["val_acc"] != "":
                    val_acc_list.append(float(line["val_acc"]))
        return train_loss_list, val_loss_list, val_acc_list

    @staticmethod
    def plot_losses(train_loss_list: list[float], val_loss_list: list[float]) -> None:
        fig = go.Figure()

        # Training losses
        fig.add_trace(
            go.Scatter(
                x=list(range(1, len(train_loss_list) + 1)),
                y=train_loss_list,
                mode="lines+markers",
                name="Training",
            )
        )

        # Validation losses
        fig.add_trace(
            go.Scatter(
                x=list(range(1, len(val_loss_list) + 1)),
                y=val_loss_list,
                mode="lines+markers",
                name="Validation",
            )
        )

        fig.update_layout(
            title="Loss vs. No. of epochs", xaxis_title="Epoch", yaxis_title="Loss"
        )

        os.makedirs("results", exist_ok=True)
        fig.write_image(os.path.join("results", "loss_graph.png"))

    @staticmethod
    def plot_scores(val_acc_list: list[float]) -> None:
        fig = go.Figure()

        # Validation accuracy
        fig.add_trace(
            go.Scatter(
                x=list(range(1, len(val_acc_list) + 1)),
                y=val_acc_list,
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
        os.makedirs("results", exist_ok=True)
        fig.write_image(os.path.join("results", "accuracy_graph.png"))


if __name__ == "__main__":
    train_loss, val_loss, val_acc = Utils.get_metrics(
        "lightning_logs/version_0/metrics.csv"
    )

    Utils.plot_losses(train_loss, val_loss)
    Utils.plot_scores(val_acc)
