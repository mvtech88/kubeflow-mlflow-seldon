from kfp import dsl
from typing import NamedTuple
from kfp.dsl import Dataset,Output,Artifact,OutputPath,InputPath, Model,Input

@dsl.component(base_image='mohitverma1688/model_train_component:v0.1',
               target_image='mohitverma1688/model_eval_component:v0.8',
               packages_to_install=['mlflow','GitPython']
               )
def predict_on_sample_image(test_dir: str, 
                         model_info: dict, 
                         image_path: str,
                         aws_access_key_id: str, 
                         aws_secret_access_key: str ,
                         ) -> NamedTuple('outputs', [('model_uri', str),('pred_label_class', str)]):
    import torch
    import torchvision
    import utils
    import os
    import mlflow
    import json

    # Set Minio credentials in the environment
    os.environ["MLFLOW_S3_ENDPOINT_URL"] = "http://minio-service.kubeflow:9000"
    os.environ["AWS_ACCESS_KEY_ID"] = aws_access_key_id
    os.environ["AWS_SECRET_ACCESS_KEY"] = aws_secret_access_key

    
    # Set up the class names, below code uses utis.py to get the class_names from the dataset
    
    DATA_DIR = test_dir 
    class_names = utils.class_names_found(DATA_DIR)
    #print(class_names)
     
    #class_names = ["pizza", "steak", "sushi"]
    
    # Set up device   
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    
    artifact_path = model_info["artifact_path"]
    artifact_uri = model_info["artifact_uri"]

    # Loading the model from the mlFlow artifact repository 
    
    mlflow.set_tracking_uri("http://my-mlflow.kubeflow:5000")
    model_uri = f"{artifact_uri}/{artifact_path}"
    model = mlflow.pytorch.load_model(model_uri)

    # Below section is to use the the random path if the image path is not provided. A random image from test data in dataset is selected for prediction. 
    # Make sure that the downloaded data has the path for testing some images for example "/data/predict/123.jpg"
    
    if image_path == None:
    
        test_data_paths =  list(Path("/data/test").glob("*/*.jpg"))
        image_path = random.choice(test_data_paths)
        print(image_path)

    # Load in the image and turn it into torch.float32 ( same type as model)
    image = torchvision.io.read_image(image_path).type(torch.float32)

    # Preprocess the image to get between 0 and 1
    image = image / 255.

    # Resize the image to be of same size as model.
    transform = torchvision.transforms.Resize(size=(64,64))
    image = transform(image)

    # Predict on image
    model.eval()
    with torch.inference_mode():
        # Put image on the model
        image = image.to(device)

        # Get the pred_logits
        pred_logits = model(image.unsqueeze(dim=0))  # Adding a new dimension for the batch size.

        # Get the pred probs
        pred_prob = torch.softmax(pred_logits, dim=1)

        # Get the pred_labels
        pred_label = torch.argmax(pred_prob, dim=1)
        pred_label_class = class_names[pred_label]

    print(f"[INFO] Pred class: {pred_label_class}, Pred_prob: {pred_prob.max():.3f}")
    pred_prob_max = pred_prob.max()
    print(type(pred_prob_max))

    outputs = NamedTuple("outputs", model_uri=str, pred_label_class=str)
    
    return outputs(model_uri, pred_label_class)
