import os
import torch
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from werkzeug.utils import secure_filename
from torchvision import transforms
from PIL import Image
from model import CrowdCounterModel


app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "http://localhost"}})

# Configure upload folder
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Define transformation
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Load models
model_dense = CrowdCounterModel().cuda()
model_sparse = CrowdCounterModel().cuda()

checkpoint_dense = torch.load("1model_best.pth.tar", map_location="cuda", weights_only=True)
checkpoint_sparse = torch.load("2model_best.pth.tar", map_location="cuda", weights_only=True)

model_dense.load_state_dict(checkpoint_dense['state_dict'])
model_sparse.load_state_dict(checkpoint_sparse['state_dict'])

model_dense.eval()
model_sparse.eval()

@app.route('/')
def home():
    return render_template('index.html')  # Create a simple HTML form to upload images

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({'error': 'No image file uploaded.'}), 400
    if 'density' not in request.form:
        return jsonify({'error': 'Density type not specified.'}), 400

    file = request.files['image']
    density_type = request.form['density']
    filename = secure_filename(file.filename)

    if filename == '':
        return jsonify({'error': 'No selected file.'}), 400

    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(filepath)

    try:
        # Select model based on density type
        if density_type == 'dense':
            model = model_dense
        elif density_type == 'sparse':
            model = model_sparse
        else:
            return jsonify({'error': 'Invalid density type. Choose "dense" or "sparse".'}), 400

        # Process image
        img = transform(Image.open(filepath).convert('RGB')).cuda()
        output = model(img.unsqueeze(0))
        predicted_count = int(output.detach().cpu().sum().numpy())

        # Clean up uploaded file
        os.remove(filepath)

        return jsonify({'predicted_count': predicted_count})

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)