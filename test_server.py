import requests
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("image_path", help="Path to image")
parser.add_argument("--url", required=True, help="Model URL")
parser.add_argument("--api_key", required=True, help="API Key")
args = parser.parse_args()

headers = {"Authorization": f"Bearer {args.api_key}"}
files = {"image": open(args.image_path, "rb")}

response = requests.post(args.url + "/predict", files=files, headers=headers)

if response.ok:
    print("Prediction:", response.json())
else:
    print("Error:", response.text)


# python test_server.py test_images/tench.jpg --url https://your-model-url --api_key your_api_key
