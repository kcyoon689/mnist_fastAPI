import os
import argparse
from io import BytesIO
from typing import Optional

import mlflow
import onnx
import uvicorn
import torch
import numpy as np
from PIL import Image
from fastapi import FastAPI, HTTPException, File, UploadFile
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from modules import MNISTDataModule, MNISTModel, Utils


app = FastAPI()
mlflow.set_tracking_uri(f"http://localhost:5000")


class TrainRequest(BaseModel):
    learning_rate: float = 0.01
    max_epochs: int = 10
    batch_size: int = 256
    other_hyperparameters: Optional[dict] = None


class RegisterRequest(BaseModel):
    run_id: str
    artifact_path: str = "model"
    registered_model_name: str = "mnist_model"
    registered_artifact_path: str = "onnx_model"


@app.get("/")
def root():
    return JSONResponse(content={"Hello": "World!"}, status_code=200)


@app.post("/train")
async def post_train(train_request: TrainRequest):
    Utils.setup_logging()

    # 학습 파라미터
    lr = train_request.learning_rate
    max_epochs = train_request.max_epochs
    batch_size = train_request.batch_size
    # other_params = train_request.other_hyperparameters or {}

    if max_epochs > 15:
        return HTTPException(
            status_code=500, detail="Exceeding max_epoch, set epoch to 15 or less"
        )

    if mlflow.active_run():
        mlflow.end_run()

    mlflow.enable_system_metrics_logging()

    mlflow.pytorch.autolog()

    dm = MNISTDataModule(batch_size)
    model = MNISTModel(
        *dm.dims,
        num_classes=dm.num_classes,
        batch_size=batch_size,
        learning_rate=lr,
        max_epochs=max_epochs,
    )
    trainer = model.trainer

    try:
        # Train the model
        with mlflow.start_run() as mlflow_run:
            trainer.fit(model=model, datamodule=dm)
            mlflow_run_dict = mlflow_run.to_dictionary()

        return JSONResponse(
            content={
                "run_id": mlflow_run_dict["info"]["run_id"],
                "artifact_path": "model",
            },
            status_code=200,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e


@app.post("/register")
async def post_register(register_request: RegisterRequest):
    run_id = register_request.run_id
    artifact_path = register_request.artifact_path
    registered_model_name = register_request.registered_model_name
    registered_artifact_path = register_request.registered_artifact_path

    try:
        model = mlflow.pytorch.load_model(f"runs:/{run_id}/{artifact_path}")
        input_sample = torch.randn((1, 1, 28, 28))
        os.makedirs("weights", exist_ok=True)
        model.to_onnx("weights/model.onnx", input_sample, export_params=True)
        onnx_model = onnx.load("weights/model.onnx")
        onnx.checker.check_model(onnx_model)

        registered_model_info = mlflow.onnx.log_model(
            onnx_model,
            registered_artifact_path,
            registered_model_name=registered_model_name,
        )

        print("registered_run_id:", registered_model_info.run_id)

        return JSONResponse(
            content={
                "registered_run_id": registered_model_info.run_id,
                "registered_artifact_path": registered_model_info.artifact_path,
            },
            status_code=200,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e


@app.post("/predict")
async def post_predict(file: UploadFile = File(...)):
    contents = await file.read()
    try:
        upload_image = Image.open(BytesIO(contents))
        image_tensor = MNISTDataModule.predict_transform(upload_image).unsqueeze(0)  # type: ignore

        client = mlflow.MlflowClient()
        for rm in client.search_registered_models():
            if rm.name == "mnist_model":
                if rm.latest_versions:
                    latest_model_run_id = rm.latest_versions[0].run_id
                    latest_model_artifact_path = rm.latest_versions[0].source.split(
                        "/"
                    )[-1]
        loaded_onnx_model = mlflow.pyfunc.load_model(
            f"runs:/{latest_model_run_id}/{latest_model_artifact_path}"
        )
        raw_onnx_outputs = loaded_onnx_model.predict(image_tensor.numpy())
        onnx_outputs = list(raw_onnx_outputs.values())[0][0]
        prediction = np.argmax(onnx_outputs)
        confidence = Utils.softmax(onnx_outputs)

        return JSONResponse(
            content={
                "label": str(prediction),
                "confidence": str(confidence),
            },
            status_code=200,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Model Serving API Service.")
    parser.add_argument(
        "--host_ip",
        type=str,
        default="localhost",
        help="host ip address",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    uvicorn.run(app, host=args.host_ip, port=8000)
