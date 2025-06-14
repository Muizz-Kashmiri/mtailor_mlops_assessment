import torch
from PIL import Image
import sys
from pytorch_model import Classifier, BasicBlock

def main(image_path):
    # Load model
    model = Classifier(BasicBlock, [2, 2, 2, 2])
    model.load_state_dict(torch.load('pytorch_model_weights.pth', map_location=torch.device('cpu')))
    model.eval()

    # Load and preprocess image
    img = Image.open(image_path).convert('RGB')
    input_tensor = model.preprocess_numpy(img).unsqueeze(0)  # Shape: [1, 3, 224, 224]

    # Predict
    with torch.no_grad():
        output = model(input_tensor)
        predicted_class = torch.argmax(output, dim=1).item()

    print(f"✅ Predicted class ID: {predicted_class}")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python test_pytorch_model.py <path_to_image>")
        sys.exit(1)

    main(sys.argv[1])
