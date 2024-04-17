import numpy as np
import mlflow
from PIL import Image
from data_module import MNISTDataModule
from utils import Utils


image = Image.open(f"samples/img_2.jpg")
image_tensor = MNISTDataModule().predict_transform(image).unsqueeze(0)

client = mlflow.MlflowClient()
for rm in client.search_registered_models():
    if rm.name == "mnist_model":
        if rm.latest_versions:
            latest_model_run_id = rm.latest_versions[0].run_id
            latest_model_artifact_path = rm.latest_versions[0].source.split("/")[-1]
loaded_onnx_model = mlflow.pyfunc.load_model(
    f"runs:/{latest_model_run_id}/{latest_model_artifact_path}"
)
raw_onnx_outputs = loaded_onnx_model.predict(image_tensor.numpy())
onnx_outputs = list(raw_onnx_outputs.values())[0][0]
prediction = np.argmax(onnx_outputs)
confidence = Utils.softmax(onnx_outputs)

print(onnx_outputs)
print(prediction)
print(confidence)
