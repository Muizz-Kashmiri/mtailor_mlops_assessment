from flask import Flask, request, jsonify
from model import OnnxModel, Preprocessor
import os
from PIL import Image

app = Flask(__name__)

# Define your valid API key
VALID_API_KEY = os.getenv("API_KEY")

# Initialize the model and preprocessor
model = OnnxModel("model.onnx")
prep = Preprocessor()

@app.before_request
def check_api_key():
    """
    This function runs before every request to check if the API key is correct.
    """
    api_key = request.headers.get("Authorization")
    if not api_key or api_key != f"Bearer {VALID_API_KEY}":
        return jsonify({"error": "Invalid or missing API key"}), 401

@app.route("/predict", methods=["POST"])
def predict():
    if 'image' not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    try:
        # Get the uploaded image from the request
        image_file = request.files['image']

        # Convert the image to a PIL image
        img = Image.open(image_file)

        # Use the updated preprocessor that can handle PIL images
        processed_img = prep.run(img)

        # Predict the class ID using the ONNX model
        prediction = model.predict(processed_img)

        # Return prediction as JSON response
        return jsonify({"predicted_class_id": prediction})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=8000)