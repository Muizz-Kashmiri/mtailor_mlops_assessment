import onnxruntime as ort
from torchvision import transforms
from PIL import Image
import numpy as np
import torch

class Preprocessor:
    def __init__(self):
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])

    def run(self, image_input):
        """
        Process either a file path (string) or PIL Image object
        """
        if isinstance(image_input, str):
            # If it's a string, treat it as a file path
            image = Image.open(image_input).convert('RGB')
        elif isinstance(image_input, Image.Image):
            # If it's already a PIL Image, use it directly
            image = image_input.convert('RGB')
        else:
            raise ValueError("Input must be either a file path (string) or PIL Image object")
        
        image = self.transform(image).unsqueeze(0)  # Add batch dimension
        return image.numpy()

class OnnxModel:
    def __init__(self, model_path="model.onnx"):
        self.session = ort.InferenceSession(model_path)
        self.input_name = self.session.get_inputs()[0].name

    def predict(self, image_array):
        outputs = self.session.run(None, {self.input_name: image_array})
        probs = outputs[0]
        return int(np.argmax(probs))