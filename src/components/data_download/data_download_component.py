from kfp import dsl

@dsl.component(base_image='python:3.10-slim',
               target_image='mohitverma1688/data-download_component:v0.4',
               packages_to_install=['pathlib','requests','kfp-kubernetes'])



def dataset_download(url: str, base_path:str, 
                     ):

    import os
    import requests
    import zipfile
    from pathlib import Path

    #Import the enviornment variable 

    #os.environ["MLFLOW_S3_ENDPOINT_URL"] = "http://minio-service.kubeflow:9000"
    #os.environ["AWS_ACCESS_KEY_ID"] = aws_access_key_id
    #os.environ["AWS_SECRET_ACCESS_KEY"] = aws_secret_access_key
    
    #s3 = boto3.client(
    #    "s3",
    #    endpoint_url="http://minio-service.kubeflow:9000",
    #    aws_access_key_id=aws_access_key_id,
    #    aws_secret_access_key=aws_secret_access_key,
    #    config=Config(signature_version="s3v4"),
    #)
    # Create data bucket if it does not yet exist
    #response = s3.list_buckets()
    #input_bucket_exists = False
    #for bucket in response["Buckets"]:
    #    if bucket["Name"] == input_bucket:
    #        input_bucket_exists = True
    #        
    #if not input_bucket_exists:
    #    s3.create_bucket(ACL="public-read-write", Bucket=input_bucket)


    # Save zip files to S3 import_bucket
    data_path = Path(base_path)
    
    if data_path.is_dir():
      print(f"{data_path} directory exists.")
    else:
      print(f"Did not find {data_path} directory, creating one...")
      data_path.mkdir(parents=True,exist_ok=True)


    # Download pizza , steak and sushi data and upload to the s3 bucket. This is example code to save the downloaded data to the bucket
    with open(f"{data_path}/data.zip", "wb") as f:
        request = requests.get(f"{url}")
        print(f"Downloading data from {url}...")
        f.write(request.content)
    #    for root, dir, files in os.walk(data_path):
    #        for filename in files:
    #            local_path = os.path.join(root,filename)
    #            s3.upload_file(
    #               local_path,
    #               input_bucket,
    #               f"{local_path}",
    #               ExtraArgs={"ACL": "public-read"},
    #             )

    # unzip the data to use the data in the next step. Data will be stored in PVC.
    with zipfile.ZipFile(data_path/"data.zip", "r") as zip_ref:
      print("Unzipping data...")
      zip_ref.extractall(data_path)


