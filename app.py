from fastapi import FastAPI, UploadFile, File
from tensorflow.keras.models import load_model
import numpy as np
from PIL import Image
import io
from fastapi.responses import JSONResponse

# Initialize FastAPI app
app = FastAPI()

# Load the model (make sure this path is correct)
model_path = "simple_model.h5"  # Ensure this path is correct
model = load_model(model_path)

# Define a simple root route
@app.get("/")
def read_root():
    return {"message": "Welcome to the model prediction service!"}

# Define class labels (same as when training the model)
class_labels = ['dew', 'fogsmog', 'frost', 'glaze', 'hail', 'lightning', 'rain', 'rainbow', 'rime', 'sandstorm', 'snow']

# Helper function to preprocess the uploaded image
def preprocess_image(img: Image.Image):
    img = img.resize((128, 128))  # Resize the image to match model's input size
    img_array = np.array(img) / 255.0  # Normalize to [0, 1]
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    return img_array

# Define a route for making predictions
@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    try:
        # Read image from incoming file
        img = Image.open(io.BytesIO(await file.read()))

        # Preprocess image
        img_array = preprocess_image(img)

        # Make prediction
        predictions = model.predict(img_array)
        predicted_class_idx = np.argmax(predictions, axis=1)[0]
        predicted_class = class_labels[predicted_class_idx]

        return JSONResponse(content={"predicted_class": predicted_class, "confidence": float(np.max(predictions))})

    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=400)
