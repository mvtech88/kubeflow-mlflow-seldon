 
#Creating a python script to register the model in MLflow

from typing import Dict, NamedTuple
from kfp import dsl
from kfp.dsl import Dataset,Output,Artifact,OutputPath,InputPath, Model,Input

@dsl.component(base_image='mohitverma1688/model_train_component:v0.1',
               target_image='mohitverma1688/register_model_component:v0.30',
               packages_to_install=['mlflow','GitPython','numpy']
               )
def register_model(parameters_json_path: InputPath('Artifact'),
                   metrics_json_path: InputPath('Artifact'),
                   model_artifact_path: InputPath('Model'),
                   experiment_name: str,
                   aws_access_key_id: str,
                   aws_secret_access_key: str,
                  ) -> dict:
    
    import mlflow
    import torch
    import json
    import model_builder
    import os
    from mlflow.types import Schema, TensorSpec
    import numpy as np
    from mlflow.models import ModelSignature

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load the model
    model = model_builder.TinyVGG(input_shape=3,
                                  hidden_units=10,
                                  output_shape=3).to(device)
    
    model.load_state_dict(torch.load(model_artifact_path))
    
    # Log the model artifact in the MlFlow artifacts .

    with open('/tmp/model_builder.py', 'w') as f:
        f.write('''
import torch
from torch import nn
class TinyVGG(nn.Module):
  """Creates the TinyVGG architecture.

  Replicates the TinyVGG architecture from the CNN explainer website in PyTorch.
  See the original architecture here: https://poloclub.github.io/cnn-explainer/

  Args:
    input_shape: An integer indicating number of input channels.
    hidden_units: An integer indicating number of hidden units between layers.
    output_shape: An integer indicating number of output units.
  """
  def __init__(self, input_shape: int, hidden_units: int, output_shape: int) -> None:
      super().__init__()
      self.conv_block_1 = nn.Sequential(
          nn.Conv2d(in_channels=input_shape,
                    out_channels=hidden_units,
                    kernel_size=3,
                    stride=1,
                    padding=0),
          nn.ReLU(),
          nn.Conv2d(in_channels=hidden_units,
                    out_channels=hidden_units,
                    kernel_size=3,
                    stride=1,
                    padding=0),
          nn.ReLU(),
          nn.MaxPool2d(kernel_size=2,
                        stride=2)
      )
      self.conv_block_2 = nn.Sequential(
          nn.Conv2d(hidden_units, hidden_units, kernel_size=3, padding=0),
          nn.ReLU(),
          nn.Conv2d(hidden_units, hidden_units, kernel_size=3, padding=0),
          nn.ReLU(),
          nn.MaxPool2d(2)
      )
      self.classifier = nn.Sequential(
          nn.Flatten(),
          # Where did this in_features shape come from?
          # It's because each layer of our network compresses and changes the shape of our inputs data.
          nn.Linear(in_features=hidden_units*13*13,
                    out_features=output_shape)
      )

  def forward(self, x: torch.Tensor):
      x = self.conv_block_1(x)
      x = self.conv_block_2(x)
      x = self.classifier(x)
      return x
      # return self.classifier(self.conv_block_2(self.conv_block_1(x))) # <- leverage the benefits of operator fusion
            ''')

    # Load the parameters and metrics for mlflow to register. 
    with open(metrics_json_path) as f:
        metrics = json.load(f)
        print(metrics)
    with open(parameters_json_path) as f:
        parameters = json.load(f)
        print(parameters)

    #Define the mlflow tracking URI and experiment name
    tracking_uri = 'http://my-mlflow.kubeflow:5000'
    experiment_name = experiment_name
    mlflow.tracking.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment_name)
    reg_model_name = "CNN-TinyVG-Model"

    # Logs the enviornment variable to be used by minio
    os.environ["GIT_PYTHON_REFRESH"] = "quiet"
    os.environ["MLFLOW_S3_ENDPOINT_URL"] = "http://minio-service.kubeflow:9000"
    os.environ["AWS_ACCESS_KEY_ID"] = aws_access_key_id
    os.environ["AWS_SECRET_ACCESS_KEY"] = aws_secret_access_key

    #Define the model schema

    input_schema = Schema([TensorSpec(np.dtype(np.float32), (1, 3, 64,64))])
    output_schema = Schema([TensorSpec(np.dtype(np.float32), (1, 3))])
    signature = ModelSignature(inputs=input_schema, outputs=output_schema)
    

    # Star the mlflow server to register the model 
    with mlflow.start_run() as run:
        for metric in metrics:
            mlflow.log_metric(metric, metrics[metric])
        mlflow.log_params(parameters)

        artifact_path = "CNN-TinyVG"
        mlflow.pytorch.log_model(model,
                                 registered_model_name=reg_model_name,
                                 signature=signature,
                                 artifact_path=artifact_path,
                                code_paths=['/tmp/model_builder.py'])

        # Example snippet to log the scripted model 
        #scripted_pytorch_model = torch.jit.script(model)
        
        #mlflow.pytorch.log_model(scripted_pytorch_model, "scripted_model")
        
        run_id = mlflow.last_active_run().info.run_id
        print(f"Logged data and model in run {run_id}...")

        # Run time model uri 
        model_uri = f"runs:/{run_id}/CNN-model"
        
        #Another way of loading model at runtime
        #loaded_model = mlflow.pytorch.load_model(model_uri)
        
        return {"artifact_path": artifact_path, "artifact_uri": run.info.artifact_uri}
        

                 
