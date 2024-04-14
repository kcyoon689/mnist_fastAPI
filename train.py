import argparse
import model
import torch
from torch.optim.lr_scheduler import OneCycleLR
from utils import get_dataloader, makedirs, plotly_plot_losses, plotly_plot_scores
import datetime
from datetime import timezone, timedelta
import mlflow


def fit(epochs, lr, model, train_loader, val_loader, device, opt_func=torch.optim.SGD):
    torch.cuda.empty_cache()
    history = []
    train_loss = []
    val_loss = []
    val_acc = []
    optimizer = opt_func(model.parameters(), lr)
    scheduler = OneCycleLR(
        optimizer, max_lr=lr, epochs=epochs, steps_per_epoch=len(train_loader)
    )

    for epoch in range(epochs):
        model.train()
        train_losses = []

        for batch in train_loader:
            images, labels = batch
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            loss = model.training_step((images, labels))
            loss.backward()
            optimizer.step()
            scheduler.step()
            train_losses.append(loss.item())

        result = evaluate(model, val_loader, device)
        result["train_loss"] = sum(train_losses) / len(train_losses)

        history.append(result)

        train_loss.append(result["train_loss"])
        val_loss.append(result["val_loss"])
        val_acc.append(result["val_acc"])
        print(
            f'Epoch [{epoch+1}/{epochs}], Train Loss: {result["train_loss"]:.4f}, Val Loss: {result["val_loss"]:.4f}, Val Acc: {result["val_acc"]:.4f}'
        )

        # MLflow에 메트릭 로깅
        mlflow.log_metrics(
            {
                "train_loss": result["train_loss"],
                "val_loss": result["val_loss"],
                "val_acc": result["val_acc"],
            },
            step=epoch,
        )

    return history, train_loss, val_loss, val_acc


@torch.no_grad()
def evaluate(model, val_loader, device):
    model.eval()
    # outputs = [model.validation_step(batch.to(device)) for batch in val_loader]
    outputs = [
        model.validation_step((images.to(device), labels.to(device)))
        for images, labels in val_loader
    ]
    return model.validation_epoch_end(outputs)


def train_model(args_dict: dict):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader, val_loader, test_loader = get_dataloader(
        args_dict.get("batch_size"), args_dict.get("val_size")
    )

    if mlflow.active_run():
        mlflow.end_run()

    # MLflow 시작
    mlflow.start_run()
    mlflow.log_params(
        {
            "learning_rate": args_dict.get("lr"),
            "batch_size": args_dict.get("batch_size"),
            "validation_size": args_dict.get("val_size"),
            "epochs": args_dict.get("n_epochs"),
        }
    )

    model_instance = model.MnistModel().to(device)
    history, train_loss, val_loss, val_acc = fit(
        args_dict.get("n_epochs"),
        args_dict.get("lr"),
        model_instance,
        train_loader,
        val_loader,
        device,
    )

    plotly_plot_losses(train_loss, val_loss)
    plotly_plot_scores(val_acc)

    # test the model
    test_result = evaluate(model_instance, test_loader, device)
    print(
        f'Test Loss: {test_result["val_loss"]:.4f}, Test Acc: {test_result["val_acc"]:.4f}'
    )

    # MLflow에 테스트 결과 로깅
    mlflow.log_metrics(
        {"test_loss": test_result["val_loss"], "test_acc": test_result["val_acc"]}
    )

    # save the model
    save_path = "./weights/"
    makedirs(save_path)

    # UTC to KST(UTC+9)
    kst_time = datetime.datetime.now(timezone.utc) + timedelta(hours=9)
    current_time_kst = kst_time.strftime("%y%m%d_%H%M%S")
    file_name = f"model_{args_dict.get('n_epochs')}epochs_{current_time_kst}.pth"
    torch.save(model_instance.state_dict(), save_path + file_name)
    print(f"'{file_name}' saved!")

    # MLflow에 모델 저장 및 등록
    # model_name = "MNIST_Model"
    mlflow.pytorch.log_model(model_instance, "model")
    # mlflow.pytorch.log_model(model_instance, "model", registered_model_name=model_name)

    # MLflow 실행 종료
    mlflow.end_run()

    print("Training completed.")

    return train_loss, val_loss, val_acc


def parse_args():
    parser = argparse.ArgumentParser(description="Train a model on MNIST dataset.")
    parser.add_argument(
        "--lr",
        type=float,
        default=0.01,
        help="learning rate for training (default: 0.01)",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=128,
        help="input batch size for training (default: 128)",
    )
    parser.add_argument(
        "--val_size",
        type=int,
        default=10000,
        help="proportion of dataset to use for validation (default: 0.2)",
    )
    parser.add_argument(
        "--n_epochs",
        type=int,
        default=10,
        help="number of epochs to train (default: 10)",
    )

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    args_dict = vars(args)
    # print(args_dict)
    train_model(args_dict)
