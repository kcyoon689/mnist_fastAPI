import argparse
from io import BytesIO
from typing import Optional

import mlflow
import uvicorn
import torch
from PIL import Image
from fastapi import FastAPI, HTTPException, File, UploadFile
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from modules import (
    MnistModel,
    predict,
    train_model,
    setup_logging,
)

app = FastAPI()


class TrainRequest(BaseModel):
    learning_rate: float = 0.01
    epochs: int = 10
    batch_size: int = 128
    val_size: int = 10000  # 검증 데이터셋 크기 추가
    other_hyperparameters: Optional[dict] = None


class RegisterRequest(BaseModel):
    experiment_id: str
    mnist_model_path: str
    artifact_path: str = "mnist_model.onnx"
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
    return JSONResponse(content={"Hello": "World!"}, status_code=200)


@app.post("/train")
async def post_train(train_request: TrainRequest):
    setup_logging()

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
        with mlflow.start_run() as mlflow_run:
            train_loss, val_loss, val_acc = train_model(args, mlflow_run)
            experiment_id = mlflow_run.info.run_id

        # # 학습 과정 시각화
        # plotly_plot_losses(train_loss, val_loss)
        # plotly_plot_scores(val_acc)

        return JSONResponse(
            content={
                "experiment_id": experiment_id,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "val_acc": val_acc,
            },
            status_code=200,
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e


@app.post("/register")
async def post_register(register_request: RegisterRequest):
    experiment_id = register_request.experiment_id
    mnist_model_path = register_request.mnist_model_path
    artifact_path = register_request.artifact_path
    register_name = register_request.register_name

    try:
        # 모델을 ONNX 형식으로 변환
        model_instance = MnistModel()
        model_instance.load_state_dict(torch.load(mnist_model_path))
        model_instance.eval()

        dummy_input = torch.randn(1, 1, 28, 28)
        torch.onnx.export(model_instance, dummy_input, artifact_path)

        result = mlflow.register_model(f"runs:/{experiment_id}/model", "test_model")

        return JSONResponse(
            content={"status": "success", "result": result}, status_code=200
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e


def parse_args():
    parser = argparse.ArgumentParser(description="Model Serving API Service.")
    parser.add_argument(
        "--host_ip",
        type=str,
        default="localhost",
        help="host ip address",
    )
    return parser.parse_args()


@app.post("/predict")
async def post_predict(file: UploadFile = File(...)):
    contents = await file.read()
    try:
        loaded_image = Image.open(BytesIO(contents))
        prediction, confidence = predict(loaded_image)
        return JSONResponse(
            content={
                "label": str(prediction),
                "confidence": str(confidence),
            },
            status_code=200,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e


if __name__ == "__main__":
    args = parse_args()
    uvicorn.run(app, host=args.host_ip, port=8000)
