import requests
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("image_path", help="Path to image")
parser.add_argument("--url", required=True, help="Model URL")
args = parser.parse_args()

files = {"image": open(args.image_path, "rb")}

response = requests.post(args.url + "/predict", files=files)

if response.ok:
    print("Prediction:", response.json())
else:
    print("Error:", response.text)



# python test_server.py test_images/S0.jpeg --url http://localhost:8000
