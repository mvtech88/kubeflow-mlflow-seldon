from kfp import kubernetes
from kubernetes import client, config
import base64
from kfp import dsl
from kfp import compiler
from src.components.data_download.data_download_component import dataset_download
from src.components.model_train_cnn.model_train_component import model_train
from src.components.model_inference.model_inference_component import model_inference
from src.components.register_model.register_model_component import register_model
from src.components.model_eval.model_eval_component import predict_on_sample_image
#from src.components.model_deployment.model_deployment_component import model_serving


BASE_PATH="/data"
URL="https://github.com/mrdbourke/pytorch-deep-learning/raw/main/data/pizza_steak_sushi.zip"
#INPUT_BUCKET="datanewbucket"
NUM_EPOCHS=10
BATCH_SIZE = 32
HIDDEN_UNITS = 10
LEARNING_RATE = 0.001
MODEL_NAME = "cnn_tinyvg_v1"
MODEL_DIR = "/data/models"
EXPORT_BUCKET = "modeloutput"
TRAIN_DIR = "/data/train"
TEST_DIR = "/data/test"
IMAGE_PATH = "/data/test/pizza/1152100.jpg"
EXPERIMENT_NAME = "CNN-TinyVG-Demo-exp1"

@dsl.pipeline(name='CNN-TinyVG-Demo',
              description='This pipeline is a demo for training,evaluating and deploying Convutional Neural network',
              display_name='Kubeflow-MlFLow-Demo')



def kubeflow_pipeline(base_path: str = BASE_PATH,
                     url:str = URL,
                     batch_size:int = BATCH_SIZE,
                     train_dir:str = TRAIN_DIR,
                     test_dir:str = TEST_DIR,
                     num_epochs: int = NUM_EPOCHS,
                     hidden_units:int = HIDDEN_UNITS,
                     learning_rate:float = LEARNING_RATE,
                     model_name: str = MODEL_NAME,
                     model_dir: str = MODEL_DIR,
                     export_bucket: str = EXPORT_BUCKET,
                     image_path: str = IMAGE_PATH,
                     experiment_name: str = EXPERIMENT_NAME
                     ):
    # Load Kubernetes configuration
    config.load_kube_config()

    # Fetch the Minio credentials from the secret

    secret_name = "minio-credentials"
    secret_namespace = "kubeflow"
    secret_key_id = "AWS_ACCESS_KEY_ID"
    secret_key_access = "AWS_SECRET_ACCESS_KEY"

    v1 = client.CoreV1Api()
    secret = v1.read_namespaced_secret(secret_name, namespace=secret_namespace)


    # Convert bytes to string
    aws_access_key_id = base64.b64decode(secret.data[secret_key_id]).decode('utf-8')
    aws_secret_access_key = base64.b64decode(secret.data[secret_key_access]).decode('utf-8')
   
    pvc1 = kubernetes.CreatePVC(
        # can also use pvc_name instead of pvc_name_suffix to use a pre-existing PVC
        pvc_name='kubeflow-pvc8',
        access_modes=['ReadWriteOnce'],
        size='500Mi',
        storage_class_name='standard',
    )
    
    task1 = dataset_download(base_path=base_path,
                            url=url,
                            )
    task1.set_caching_options(False)
    
    task2 = model_train(batch_size=batch_size,
                        num_epochs=num_epochs,
                        train_dir=train_dir,
                        test_dir=test_dir,
                        hidden_units=hidden_units,
                        learning_rate=learning_rate,
                        model_name=model_name,
                        model_dir=model_dir,
                        export_bucket=export_bucket,
                        ).after(task1)
    task2.set_caching_options(False)
    
    task3 = model_inference(test_dir=test_dir,
                           model_artifact_path=task2.outputs["model_artifact_path"],
                           train_dir=train_dir,
                           learning_rate=learning_rate,
                           batch_size=batch_size,
                           num_epochs=num_epochs,
                           model_name=model_name
                           ).after(task2)
    task3.set_caching_options(False)
    
    task4 = register_model(model_artifact_path=task2.outputs["model_artifact_path"],
                           parameters_json_path=task2.outputs["parameters_json_path"],
                           metrics_json_path=task3.outputs["metrics_json_path"],
                           aws_access_key_id=aws_access_key_id,
                           aws_secret_access_key=aws_secret_access_key,                         
                           experiment_name=experiment_name).after(task3)
    task4.set_caching_options(False)

    task5 = predict_on_sample_image(test_dir=test_dir,
                                 image_path=image_path,
                                 model_info=task4.output,
                                 aws_access_key_id=aws_access_key_id,
                                 aws_secret_access_key=aws_secret_access_key).after(task4)
    task5.set_caching_options(False)

    

#    task6 = model_serving(model_uri=task5.outputs['model_uri']).after(task5)
#    task6.set_caching_options(False)
    
              
                                                  
    kubernetes.mount_pvc(
        task1,
        pvc_name=pvc1.outputs['name'],
        mount_path='/data',
    )
    kubernetes.mount_pvc(
        task2,
        pvc_name=pvc1.outputs['name'],
        mount_path='/data',
    )
    kubernetes.mount_pvc(
        task3,
        pvc_name=pvc1.outputs['name'],
        mount_path='/data',
    )
    kubernetes.mount_pvc(
        task4,
        pvc_name=pvc1.outputs['name'],
        mount_path='/data',
    )
    kubernetes.mount_pvc(
        task5,
        pvc_name=pvc1.outputs['name'],
        mount_path='/data',
    )
#    kubernetes.mount_pvc(
#        task6,
#        pvc_name=pvc1.outputs['name'],
#        mount_path='/data',
#    )
    delete_pvc1 = kubernetes.DeletePVC(pvc_name=pvc1.outputs['name']).after(task5)

compiler.Compiler().compile(kubeflow_pipeline, 'kubeflow-demo2.yaml')
