import os
import cv2
import pytesseract
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from PIL import Image

# ✅ If tesseract is installed at a specific path, add it here
# Example for Windows
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# ✅ Load your currency classifier model
model_path = "currency_classifier_model.keras"
if not os.path.exists(model_path):
    raise FileNotFoundError(f"Model file '{model_path}' not found.")
model = load_model(model_path)
print(f"[INFO] Loaded model: {model_path}")

# ✅ Load image
image_path = "test_image.jpg"
if not os.path.exists(image_path):
    raise FileNotFoundError(f"Test image '{image_path}' not found.")
img = Image.open(image_path).resize((224, 224))
img_array = img_to_array(img) / 255.0
img_array = np.expand_dims(img_array, axis=0)

# ✅ Predict using model
prediction = model.predict(img_array)
predicted_class_index = np.argmax(prediction)
class_labels = ['10', '20', '50', '100', '200', '500', '2000']  # Change this list to match your model
predicted_label = class_labels[predicted_class_index]

# ✅ OCR (Text Recognition)
ocr_image = cv2.imread(image_path)
gray = cv2.cvtColor(ocr_image, cv2.COLOR_BGR2GRAY)
ocr_result = pytesseract.image_to_string(gray)
ocr_detected = ""

# Try to find denomination in text
for label in class_labels:
    if label in ocr_result:
        ocr_detected = label
        break

# ✅ Final output
print("\n====================")
print(f"[🔍 MODEL PREDICTION] Currency note: ₹{predicted_label}")
print(f"[🔤 OCR DETECTION] Detected text: {ocr_result.strip()}")
if ocr_detected:
    print(f"[✅ OCR Match] Found ₹{ocr_detected} in image text.")
else:
    print(f"[❌ OCR Match] No clear denomination found in text.")
print("====================\n")
