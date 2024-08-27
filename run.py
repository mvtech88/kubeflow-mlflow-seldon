from kfp.client import Client

client = Client(host='http://localhost:8002')
run = client.create_run_from_pipeline_package(
    'kubeflow-demo2.yaml',
     )
