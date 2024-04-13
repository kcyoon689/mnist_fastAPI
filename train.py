import argparse
import model
import torch
from torch.optim.lr_scheduler import OneCycleLR
import plotly.graph_objects as go
from utils import get_dataloader
from utils import makedirs
from utils import plotly_plot_losses, plotly_plot_scores
import datetime
from datetime import timezone, timedelta
import mlflow


# argparse를 사용하여 명령줄 인수를 파싱
parser = argparse.ArgumentParser(description="Train a model on MNIST dataset.")
parser.add_argument(
    "--lr", type=int, default=0.01, help="learning_rate for training (default: 0.01)"
)
parser.add_argument(
    "--batch_size",
    type=int,
    default=128,
    help="input batch size for training (default: 64)",
)
parser.add_argument(
    "--val_size",
    type=int,
    default=10000,
    help="size of validation dataset (default: 10000)",
)
parser.add_argument(
    "--n_epochs", type=int, default=10, help="number of epochs to train (default: 10)"
)
args = parser.parse_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
train_loader, val_loader, test_loader = get_dataloader(args.batch_size, args.val_size)

# MLflow 실험 시작
mlflow.start_run()
mlflow.log_params(
    {
        "learning_rate": args.lr,
        "batch_size": args.batch_size,
        "validation_size": args.val_size,
        "epochs": args.n_epochs,
    }
)


def fit(epochs, lr, model, train_loader, val_loader, opt_func=torch.optim.SGD):
    torch.cuda.empty_cache()
    history = []
    train_loss = []
    val_loss = []
    val_acc = []
    optimizer = opt_func(model.parameters(), lr)
    scheduler = OneCycleLR(
        optimizer, lr, epochs=epochs, steps_per_epoch=len(train_loader)
    )

    for epoch in range(epochs):
        # Training Phase
        model.train()
        train_losses = []
        for batch in train_loader:
            images, labels = batch
            images, labels = images.to(device), labels.to(device)
            loss = model.training_step((images, labels))
            train_losses.append(loss)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            scheduler.step()

        # Validation phase
        result = evaluate(model, val_loader)
        result["train_loss"] = torch.stack(train_losses).mean().item()
        model.epoch_end(epoch, result, [scheduler.get_last_lr()[0]])
        history.append(result)
        train_loss.append(result["train_loss"])
        val_loss.append(result["val_loss"])
        val_acc.append(result["val_acc"])

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
def evaluate(model, val_loader):
    model.eval()
    val_losses = []
    for batch in val_loader:
        images, labels = batch
        images, labels = images.to(device), labels.to(
            device
        )  # 데이터를 모델의 디바이스로 이동
        output = model.validation_step(
            (images, labels)
        )  # 수정: 배치 데이터를 튜플로 전달
        val_losses.append(output)
    return model.validation_epoch_end(val_losses)


if __name__ == "__main__":
    model_instance = model.MnistModel().to(device)
    history, train_loss, val_loss, val_acc = fit(
        args.n_epochs, args.lr, model_instance, train_loader, val_loader
    )

    # plot losses and accuracy
    plotly_plot_losses(train_loss, val_loss)
    plotly_plot_scores(val_acc)

    # test the model
    result = evaluate(model_instance, test_loader)
    print(result)
    print(result["val_loss"], result["val_acc"])

    # MLflow에 테스트 결과 로깅
    mlflow.log_metrics({"test_loss": result["val_loss"], "test_acc": result["val_acc"]})

    # save the model
    save_path = "./weights/"
    makedirs(save_path)

    # UTC to KST(UTC+9)
    kst_time = datetime.datetime.now(timezone.utc) + timedelta(hours=9)
    current_time_kst = kst_time.strftime("%y%m%d_%H%M%S")
    epochs = args.n_epochs
    file_name = f"model_{epochs}epochs_{current_time_kst}.pth"
    torch.save(model_instance.state_dict(), save_path + file_name)
    print(f"'{file_name}' save!")

    # MLflow에 모델 저장
    mlflow.pytorch.log_model(model_instance, "model")

    # MLflow 실행 종료
    mlflow.end_run()

    print("Training completed.")
