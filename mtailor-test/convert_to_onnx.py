# convert_to_onnx.py

import torch
from pytorch_model import Classifier, BasicBlock

# Instantiate model
model = Classifier(BasicBlock, [2, 2, 2, 2])
model.load_state_dict(torch.load('pytorch_model_weights.pth', map_location=torch.device('cpu')))
model.eval()

# Dummy input for ONNX export
dummy_input = torch.randn(1, 3, 224, 224)

# Export to ONNX
torch.onnx.export(
    model,
    dummy_input,
    "model.onnx",
    input_names=['input'],
    output_names=['output'],
    dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}},
    opset_version=11
)

print("✅ Model successfully exported to model.onnx")
