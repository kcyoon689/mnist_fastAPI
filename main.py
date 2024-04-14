import torch
from fastapi import FastAPI, HTTPException, File, UploadFile
from pydantic import BaseModel
import uvicorn
from typing import Optional
from model import MnistModel
from train import train_model
from utils import (
    setup_logging,
    setup_experiment_tracking,
    plotly_plot_losses,
    plotly_plot_scores,
)
import mlflow.pytorch
from prediction import predict
import numpy as np
from PIL import Image
from io import BytesIO

app = FastAPI()


class TrainRequest(BaseModel):
    learning_rate: float
    epochs: int
    batch_size: int
    val_size: int = 10000  # 검증 데이터셋 크기 추가
    other_hyperparameters: Optional[dict] = None


class RegisterRequest(BaseModel):
    experiment_id: str
    model_path: str
    model_name: str = "mnist_model.onnx"
    register_name: str = "mnist_model"


# class predictRequest(BaseModel):
#     input_img_path: str
#     # mlflow model
#     model_name: str
#     # mlflow registered model
#     registered_model_name: str
#     # mlflow experiment
#     experiment_name: str


@app.get("/")
def root():
    return {"Hello": "World!"}


@app.post("/train/")
async def train(train_request: TrainRequest):
    setup_logging()
    experiment_id = setup_experiment_tracking()

    # 학습 파라미터
    lr = train_request.learning_rate
    epochs = train_request.epochs
    batch_size = train_request.batch_size
    val_size = train_request.val_size
    other_params = train_request.other_hyperparameters or {}

    try:
        # train_loader, val_loader, test_loader = get_dataloader(batch_size, val_size)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        args = {
            "lr": lr,
            "n_epochs": epochs,
            "batch_size": batch_size,
            "val_size": val_size,
            **other_params,  # 기타 하이퍼파라미터 추가
        }

        # 현재 활성화된 MLflow 실행이 있다면 종료
        if mlflow.active_run():
            mlflow.end_run()

        # 새로운 MLflow 실행 시작
        with mlflow.start_run():
            train_loss, val_loss, val_acc = train_model(args)

        # # 학습 과정 시각화
        # plotly_plot_losses(train_loss, val_loss)
        # plotly_plot_scores(val_acc)

        return {
            "experiment_id": experiment_id,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "val_acc": val_acc,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/register/")
async def register(register_request: RegisterRequest):
    experiment_id = register_request.experiment_id
    model_path = register_request.model_path
    model_name = register_request.model_name
    register_name = register_request.register_name

    try:
        # 모델을 ONNX 형식으로 변환
        model_instance = MnistModel()
        model_instance.load_state_dict(torch.load(model_path))
        model_instance.eval()

        dummy_input = torch.randn(1, 1, 28, 28)
        torch.onnx.export(model_instance, dummy_input, model_name)

        # 현재 활성화된 MLflow 실행이 있다면 종료
        if mlflow.active_run():
            mlflow.end_run()

        # 새로운 MLflow 실행 시작
        with mlflow.start_run():
            # mlflow.pytorch.log_model(model_instance, model_name)
            mlflow.pytorch.log_model(
                model_instance, model_name, registered_model_name=register_name
            )

        return {"status": "success"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# model = load_model("mnist_model.h5")


@app.post("/predict")
# define a form with a multipart input, which will be the image in this case.
async def predict(file: UploadFile = File(...)):
    contents: bytes = await file.read()
    loaded_image: Image = Image.open(BytesIO(contents))
    # loaded_image = np.expand_dims(loaded_image, axis=0)
    prediction, confidence = await predict(loaded_image)
    return {"label": str(prediction), "confidence": str(confidence)}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
