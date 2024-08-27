import os
import numpy as np
import torch
import mlflow
import model_builder
import random
import base64
import PIL.Image as Image
from io import BytesIO

class Inference(object):
    def __init__(self):
        self.device = 'cpu'
        #mlflow.tracking.set_tracking_uri("http://my-mlflow.kubeflow:5000")
        #run_id = mlflow.last_active_run().info.run_id
        #self.model_path = f"runs:/{run_id}/CNN-model"
        os.environ["MLFLOW_S3_ENDPOINT_URL"] = "http://minio-service.kubeflow:9000"
        os.environ["AWS_ACCESS_KEY_ID"] = "minio"
        os.environ["AWS_SECRET_ACCESS_KEY"] = "minio123"
        #self.model = mlflow.pytorch.load_model(self.model_path)
        self.model_uri = os.environ['MODEL_URI']
        self.model = mlflow.pytorch.load_model(self.model_uri)
        #self.model = model_builder.TinyVGG(input_shape=3,
        #                          hidden_units=10,
        #                          output_shape=3).to(self.device)
        self.class_names = ["pizza", "steak", "sushi"]
        #print(self.model_uri)

    def predict(self, X, feature_names):
        X = X[0].encode()
        im_bytes = base64.b64decode(X)
        im_file = BytesIO(im_bytes)
        img = Image.open(im_file)
        img = np.asarray(img.resize((64, 64))).astype(np.float32) / 255.0
        if img.ndim == 2:
            img = np.stack([img, img, img], axis=-1)
        img = np.transpose(img, (-1, 0, 1))
        img = np.expand_dims(img, axis=0)
        img = torch.from_numpy(img).to(self.device)
        with torch.inference_mode():
         pred_logits = self.model(img)
         pred_prob = torch.softmax(pred_logits, dim=1).numpy()
         op = torch.nn.Sigmoid()(self.model(img))
         op = torch.where(op > 0.5, 1, 0).numpy()
        return [pred_prob.tolist()]

