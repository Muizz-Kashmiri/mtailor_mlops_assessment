import onnxruntime as ort
import numpy as np
from PIL import Image
import sys
from pytorch_model import Classifier, BasicBlock
import torch
from torchvision import transforms

# Preprocessing steps (same as in your PyTorch model)
def preprocess_image(image_path):
    resize = transforms.Resize((224, 224))  # Resize image to 224x224
    crop = transforms.CenterCrop((224, 224))  # Center crop to 224x224
    to_tensor = transforms.ToTensor()  # Convert image to tensor
    normalize = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  # Normalize with ImageNet values

    # Open the image, apply preprocessing, and return as a tensor
    img = Image.open(image_path).convert('RGB')
    img = resize(img)
    img = crop(img)
    img = to_tensor(img)
    img = normalize(img)
    
    # Convert to numpy array
    return img.unsqueeze(0).numpy()  # Add batch dimension (1, 3, 224, 224)

# Inference using ONNX model
def run_inference(image_path):
    # Load ONNX model
    ort_session = ort.InferenceSession("model.onnx")

    # Prepare image for prediction
    input_data = preprocess_image(image_path)

    # Get the name of the input layer
    input_name = ort_session.get_inputs()[0].name

    # Perform inference
    outputs = ort_session.run(None, {input_name: input_data})

    # Get the predicted class (index of the maximum value)
    predicted_class = np.argmax(outputs[0])

    print(f"Predicted class ID: {predicted_class}")
    return predicted_class

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python test_onnx_model.py <path_to_image>")
        sys.exit(1)

    image_path = sys.argv[1]
    run_inference(image_path)
