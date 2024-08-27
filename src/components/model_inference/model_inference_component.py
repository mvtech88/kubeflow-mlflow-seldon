
# Creating a python script to predict a model

from typing import Dict, NamedTuple
from kfp import dsl
from kfp.dsl import Dataset,Output,Artifact,OutputPath,InputPath, Model,Input,HTML

@dsl.component(base_image='mohitverma1688/model_train_component:v0.1',
               target_image='mohitverma1688/model_inference_component:v0.20',
               packages_to_install=['pandas','matplotlib']
               )

def model_inference(model_artifact_path: InputPath('Model'),
                    num_epochs: int,
                    batch_size:int,
                    learning_rate: float,
                    train_dir: str,
                    test_dir: str,
                    model_name: str,
                    test_loss: Output[HTML],
                    metrics_json_path: OutputPath('Artifact')
                    ) -> Dict[str, str]:

    import torch
    import model_builder, utils, model_inference, data_setup
    import json
    import os
    import pandas as pd
    from torchvision import transforms
    from io import BytesIO
    import base64
    import matplotlib.pyplot as plt
    

    # Setup hyperparameters
    NUM_EPOCHS = num_epochs
    BATCH_SIZE = batch_size
    LEARNING_RATE = learning_rate

    # Setup directories
    TRAIN_DIR = train_dir
    TEST_DIR = test_dir


    # Create transforms
    data_transform = transforms.Compose([
      transforms.Resize((64, 64)),
      transforms.ToTensor()
    ])
    # Create DataLoaders with help from data_setup.py
    train_dataloader, test_dataloader, class_names = data_setup.create_dataloaders(
        train_dir=TRAIN_DIR,
        test_dir=TEST_DIR,
        transform=data_transform,
        batch_size=BATCH_SIZE
    )  
    
    # Set up device   
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Load the model
    model = model_builder.TinyVGG(input_shape=3,
                                  hidden_units=10,
                                  output_shape=3).to(device)
    
    model.load_state_dict(torch.load(model_artifact_path))

    # Set loss and optimizer
    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(),
                                lr=LEARNING_RATE)

    # Start training with help from engine.py
    test_acc_last_epoch,test_loss_last_epoch,results = model_inference.test_result(model=model,
                       test_dataloader=test_dataloader,
                       loss_fn=loss_fn,
                       epochs=NUM_EPOCHS,
                       device=device)
    
    df = pd.DataFrame(results)
    #Get number of epochs

    epochs = range(len(df))
        

    # Plot train loss
    tmpfile = BytesIO()
    plt.figure(figsize=(15,10))
    plt.subplot(2,2,1)
    plt.plot(epochs, df["test_loss"], label={model_name})
    plt.title("Test Loss")
    plt.xlabel("Epochs")
    plt.legend()
    plt.savefig(tmpfile, format="png")
    tmpfile.seek(0)
    encoded = base64.b64encode(tmpfile.read()).decode("utf-8")
    test_loss.path = f"{test_loss.path}.html"
    html = f"<img src='data:image/png;base64,{encoded}'>"
    with open(test_loss.path, 'w') as f:
        f.write(html)
    

    # Below output parameters will be used by the mlflow to log the model in the next step.

    with open(metrics_json_path, 'w') as f:
        json.dump({'accuracy': test_acc_last_epoch, 'loss': test_loss_last_epoch}, f)
        
    return {"model_name" : model_name, 
            "test_acc_last_epoch": test_acc_last_epoch,
            "test_loss_last_epoch": test_loss_last_epoch}
