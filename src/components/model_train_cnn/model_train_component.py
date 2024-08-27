from kfp import dsl
from kfp import compiler
from typing import Dict
from kfp.dsl import Dataset,Output,Artifact,OutputPath,InputPath, Model,HTML

@dsl.component(base_image='mohitverma1688/model_train_component:v0.4',
               target_image='mohitverma1688/model_train_component:v0.24',
               packages_to_install=['pandas','matplotlib']
               )

def model_train(num_epochs:int, 
                batch_size:int, 
                hidden_units:int,
                learning_rate: float,
                train_dir: str,
                test_dir: str,
                model_name: str,
                model_dir: str,
                model_artifact_path: OutputPath('Model'),
                parameters_json_path: OutputPath('Artifact'),
                train_loss: Output[HTML],
                export_bucket: str = "modelbucket", 
               ) -> Dict[str, list] :

            import os
            import json
            import pandas as pd
            import torch
            import data_setup, model_train, model_builder, utils
            from io import BytesIO
            import base64
            import matplotlib.pyplot as plt

            from torchvision import transforms
         
            

            # Setup hyperparameters
            NUM_EPOCHS = num_epochs
            BATCH_SIZE = batch_size
            HIDDEN_UNITS = hidden_units
            LEARNING_RATE = learning_rate
            MODEL_NAME = model_name
            MODEL_DIR = model_dir
            EXPORT_BUCKET = export_bucket

    

            # Setup directories
            TRAIN_DIR = train_dir
            TEST_DIR = test_dir

            # Setup target device
            device = "cuda" if torch.cuda.is_available() else "cpu"

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

            # Create model with help from model_builder.py
            model = model_builder.TinyVGG(
                input_shape=3,
                hidden_units=HIDDEN_UNITS,
                output_shape=len(class_names)
            ).to(device)

            # Set loss and optimizer
            loss_fn = torch.nn.CrossEntropyLoss()
            optimizer = torch.optim.Adam(model.parameters(),
                                         lr=LEARNING_RATE)

            # Start training with help from engine.py
            result = model_train.train(model=model,
                         train_dataloader=train_dataloader,
                         loss_fn=loss_fn,
                         optimizer=optimizer,
                         epochs=NUM_EPOCHS,
                         device=device)
    
             

            df = pd.DataFrame(result)
    
            # Example Saving dataset as an artifact in kubeflow pipeline. 
            #output_dataset.path = f"{output_dataset.path}.csv"
            #df.to_csv(output_dataset.path, index=False)
            #with open(output_dataset.path, "w") as file:
            #file.write(df.to_csv(index=False))
        

            #Get number of epochs

            epochs = range(len(df))
        

            # Plot train loss
            tmpfile = BytesIO()
            plt.figure(figsize=(15,10))
            plt.subplot(2,2,1)
            plt.plot(epochs, df["train_loss"], label={model_name})
            plt.title("Train Loss")
            plt.xlabel("Epochs")
            plt.legend()
            plt.savefig(tmpfile, format="png")
            tmpfile.seek(0)
            encoded = base64.b64encode(tmpfile.read()).decode("utf-8")
            train_loss.path = f"{train_loss.path}.html"
            html = f"<img src='data:image/png;base64,{encoded}'>"
            with open(train_loss.path, 'w') as f:
                f.write(html)

                
                
            # Save the model with help from utils.py on the local PVC and if export_bucket is provided then on minio storage also.
            utils.save_model(model=model,
                             model_dir=MODEL_DIR,
                             model_name=MODEL_NAME + ".pth",
                             export_bucket=EXPORT_BUCKET)

            print("saving to kfp now")
        
            # Saving the model as kfp output artifcat. 
    
            torch.save(model.state_dict(),
                       model_artifact_path)

            # Below steps logs the parameters to be used in the next step to be used by mlflow server
            with open(parameters_json_path, 'w') as f:
                json.dump({'lr': learning_rate, 'batch_size': batch_size, 'epochs': num_epochs}, f)
        
            # return the dictionary , 
            return result
