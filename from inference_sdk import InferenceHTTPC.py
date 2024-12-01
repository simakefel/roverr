# import the inference-sdk
from inference_sdk import InferenceHTTPClient

# initialize the client
CLIENT = InferenceHTTPClient(
    api_url="https://detect.roboflow.com",  # Roboflow API URL
    api_key="D6O6tzpNXel87uTZPjNc",  # API key
)

# infer on a local image (tam dosya yoluyla)
result = CLIENT.infer(
    "C:/Users/Sima/OneDrive/Belgeler/photo-1727156275339-aad186798856.jpg",
    model_id="stop-sign-ocamr/1",
)  # Model ID ve yerel görüntü adı

# sonuçları yazdır
print(result)
