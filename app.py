from flask import Flask, request, render_template
import os, base64
from PIL import Image
from io import BytesIO

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 10 * 1024 * 1024  # Allow up to 10MB uploads

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    data_url = request.form['image_data']
    header, encoded = data_url.split(',', 1)
    img_data = base64.b64decode(encoded)

    # Save image temporarily
    image_path = os.path.join('static', 'uploads', 'temp.jpg')
    os.makedirs(os.path.dirname(image_path), exist_ok=True)
    with open(image_path, 'wb') as f:
        f.write(img_data)

    # ✅ Your prediction logic here
    prediction_result = "₹500 - Indian Rupee"  # Dummy result
    return prediction_result

if __name__ == '__main__':
    app.run(debug=True)
