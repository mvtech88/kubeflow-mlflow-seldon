"""
Contains various utility functions for PyTorch model training and saving.
"""
import torch
from pathlib import Path
import os
import boto3
from botocore.client import Config


def class_names_found(DATA_DIR):
    """Fetch the class_names from the data directory on the train dataset."""
    data_path = Path(DATA_DIR)
    target_directory = data_path
    print(f"Data directory: {target_directory}")
    # Get the class names from the target directory.
    class_names_found = sorted([entry.name for entry in list(os.scandir(target_directory))])
    return class_names_found
 



def save_model(model: torch.nn.Module,
               model_dir: str,
               model_name: str,
               export_bucket: str = None):
    
  """Saves a PyTorch model to a target directory.

  Args:
    model: A target PyTorch model to save.
    target_dir: A directory for saving the model to.
    model_name: A filename for the saved model. Should include
      either ".pth" or ".pt" as the file extension.

  Example usage:
    save_model(model=model_0,
               model_dir="models",
               model_name="05_going_modular_tingvgg_model.pth"
               )
  """


  # Create target directory
  target_dir_path = Path(model_dir)
  target_dir_path.mkdir(parents=True,
                        exist_ok=True)
  # Create model save path
  assert model_name.endswith(".pth") or model_name.endswith(".pt"), "model_name should end with '.pt' or '.pth'"
  model_save_path = target_dir_path / model_name
  # Save the model state_dict()
  print(f"[INFO] Saving model to: {model_save_path}")
  torch.save(obj=model.state_dict(),
             f=model_save_path)
    
  if export_bucket:
      # Save the model in the s3 Bucket also
      s3 = boto3.client(
        "s3",
        endpoint_url="http://minio-service.kubeflow:9000",
        aws_access_key_id="minio",
        aws_secret_access_key="minio123",
        config=Config(signature_version="s3v4"),
      )
      # Create export bucket if it does not yet exist
      response = s3.list_buckets()
      export_bucket_exists = False
      for bucket in response["Buckets"]:
          if bucket["Name"] == export_bucket:
              export_bucket_exists = True

      if not export_bucket_exists:
          s3.create_bucket(ACL="public-read-write", Bucket=export_bucket)
          print("export bucket created")

          
      

      for root, dirs, files in os.walk(model_dir):
          for filename in files:
              local_path = os.path.join(root, filename)
              s3.upload_file(
                local_path,
                export_bucket,
                f"models/{model_name}",
                ExtraArgs={"ACL": "public-read"},
            )
      response = s3.list_objects(Bucket=export_bucket)
      print(f"All objects in {export_bucket}:")
      for file in response["Contents"]:
          print("{}/{}".format(export_bucket, file["Key"]))

def accuracy_fn(y_true, y_pred):
    """Calculates accuracy between truth labels and predictions.

    Args:
        y_true (torch.Tensor): Truth labels for predictions.
        y_pred (torch.Tensor): Predictions to be compared to predictions.

    Returns:
        [torch.float]: Accuracy value between y_true and y_pred, e.g. 78.45
    """
    correct = torch.eq(y_true, y_pred).sum().item()
    acc = (correct / len(y_pred)) * 100
    return acc
          
      
      

    
 
