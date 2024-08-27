from kfp import dsl
@dsl.component(
    packages_to_install=["kserve==0.12.0","ray[serve]<=2.9.3,>=2.9.2"],
    base_image="python:3.10",
)
def model_serving(model_uri: str):
    from kubernetes import client 
    from kserve import KServeClient
    from kserve import constants
    from kserve import utils
    from kserve import V1beta1InferenceService
    from kserve import V1beta1InferenceServiceSpec
    from kserve import V1beta1PredictorSpec
    from kserve import V1beta1TorchServeSpec
    from kserve import V1beta1ModelSpec
    from kserve import V1beta1ModelFormat
    import os

    namespace = utils.get_default_target_namespace()
    
    name='cnn-tinyvg-v1'
    kserve_version='v1beta1'
    api_version = constants.KSERVE_GROUP + '/' + kserve_version

    isvc = V1beta1InferenceService(api_version=api_version,
                                   kind=constants.KSERVE_KIND,
                                   metadata=client.V1ObjectMeta(
                                       name=name, namespace=namespace, annotations={'sidecar.istio.io/inject':'false'}),
                                   spec=V1beta1InferenceServiceSpec(
                                   predictor=V1beta1PredictorSpec( model=V1beta1ModelSpec(model_format=V1beta1ModelFormat(name='mlflow')),
                                   service_account_name='mlflow-sa',
                                   pytorch=(V1beta1TorchServeSpec(storage_uri=model_uri)))))
    KServe = KServeClient()
    KServe.create(isvc)
    
