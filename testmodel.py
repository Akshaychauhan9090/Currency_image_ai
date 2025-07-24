from PIL import Image
import numpy as np
from tensorflow.keras.models import load_model

# Step 1: Load your trained model
model = load_model("currency_model.keras")

# Step 2: Load and preprocess the image
img = Image.open("test_image.jpg").resize((224, 224))
img_array = np.array(img) / 255.0
img_array = np.expand_dims(img_array, axis=0)

# Step 3: Predict
prediction = model.predict(img_array)

# Step 4: Class labels (update if needed)
class_names = ['10', '20', '50', '100', '200', '500', '2000']

# Step 5: Show result
print("Predicted:", class_names[np.argmax(prediction)])
