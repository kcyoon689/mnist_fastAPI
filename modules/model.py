import mlflow
import os
import onnx
import onnxruntime
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torchmetrics.functional import accuracy
import lightning as L
from lightning.pytorch.callbacks import ModelSummary
from lightning.pytorch.utilities.types import OptimizerLRSchedulerConfig
from modules.data_module import MNISTDataModule

PATH_WEIGHTS = os.getenv("PATH_WEIGHTS", "weights")
os.makedirs(PATH_WEIGHTS, exist_ok=True)


class MNISTModel(L.LightningModule):
    def __init__(
        self,
        channels,
        width,
        height,
        num_classes,
        hidden_size=64,
        learning_rate=0.01,
        max_epochs=10,
    ):
        super().__init__()
        self.example_input_array = torch.Tensor(hidden_size, 1, 28, 28)
        self.save_hyperparameters()

        # We take in input dimensions as parameters and use those to dynamically build model.
        self.channels = channels
        self.width = width
        self.height = height
        self.num_classes = num_classes
        self.hidden_size = hidden_size
        self.learning_rate = learning_rate

        self.trainer = L.Trainer(
            accelerator="auto",
            devices=1,
            callbacks=[ModelSummary(max_depth=-1)],
            max_epochs=max_epochs,
            max_time={
                "minutes": 10,
            },
            num_sanity_val_steps=2,
            # profiler="simple",
        )

        self.model = nn.Sequential(
            nn.Flatten(),
            nn.Linear(channels * width * height, hidden_size),
            nn.ReLU(),
            # nn.Dropout(0.1),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            # nn.Dropout(0.1),
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
    mlflow.pytorch.autolog()

    dm = MNISTDataModule()
    model = MNISTModel(*dm.dims, num_classes=dm.num_classes)
    trainer = model.trainer

    # Train the model
    with mlflow.start_run() as mlflow_run:
        trainer.fit(model=model, datamodule=dm)
        mlflow_run_dict = mlflow_run.to_dictionary()
        print(mlflow_run_dict)

    # Test the model
    trainer.test(model=model, datamodule=dm)

    # Predict
    print("answer: [2, 0, 9, 0, 3, 7, 0, 3, 0, 3]")
    results = []
    for idx in range(1, 11):
        image = Image.open(f"samples/img_{idx}.jpg")
        image_tensor = MNISTDataModule().predict_transform(image).unsqueeze(0)
        torch_outputs = model(image_tensor.to(model.device))
        confidence = F.softmax(torch_outputs, dim=1)[0] * 100
        result = torch.argmax(torch_outputs, dim=1)[0]
        # print(torch_outputs[0].detach().numpy())
        # print(confidence.detach().numpy())
        # print(result.item())
        results.append(result.item())
    print(results)

    # convert to onnx model
    # input_sample = torch.randn((1, 1, 28, 28))  # image_tensor
    model.to_onnx("weights/model.onnx", image_tensor, export_params=True)
    onnx_model = onnx.load("weights/model.onnx")
    onnx.checker.check_model(onnx_model)

    model_info = mlflow.onnx.log_model(
        onnx_model,
        "onnx_model",
        registered_model_name="test_test",
    )
    print(model_info.artifact_path)
    print(model_info.flavors)
    print(model_info.metadata)
    print(model_info.mlflow_version)
    print(model_info.model_uri)
    print(model_info.run_id)
    print(model_info.saved_input_example_info)
    print(model_info.signature_dict)
    print(model_info.utc_time_created)
    print(torch_outputs)
    print(torch_outputs[0].detach().numpy())

    ort_session = onnxruntime.InferenceSession("weights/model.onnx")
    onnx_outputs = ort_session.run(
        None, {ort_session.get_inputs()[0].name: image_tensor.numpy()}
    )
    print(onnx_outputs)
    print(onnx_outputs[0][0])

    # Load the logged model and make a prediction
    loaded_onnx_model = mlflow.pyfunc.load_model(model_info.model_uri)
    loaded_onnx_outputs = loaded_onnx_model.predict(image_tensor.numpy())
    print(loaded_onnx_outputs)
    print(list(loaded_onnx_outputs.values())[0][0])

    # Listing registered model
    client = mlflow.MlflowClient()
    print(client.search_registered_models())
    for rm in client.search_registered_models():
        if rm.name == "test_test":
            print(rm.latest_versions)
            if rm.latest_versions:
                print(rm.latest_versions[0])
                print(rm.latest_versions[0].run_id)
                print(rm.latest_versions[0].source)

    # Compare the PyTorch results with the ones from the ONNX Runtime
    assert len(torch_outputs) == len(onnx_outputs)
    torch.testing.assert_close(
        torch_outputs[0].detach(),
        torch.tensor(onnx_outputs[0][0]),
    )

    assert len(torch_outputs) == len(loaded_onnx_outputs)
    torch.testing.assert_close(
        torch_outputs[0].detach(),
        torch.tensor(list(loaded_onnx_outputs.values())[0][0]),
    )

    print("PyTorch and ONNX Runtime output matched!")
    print(f"Output length: {len(loaded_onnx_outputs)}")
    print(f"Sample output: {loaded_onnx_outputs}")
