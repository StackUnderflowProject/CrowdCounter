import os
import torch
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from werkzeug.utils import secure_filename
from torchvision import transforms
from PIL import Image
from model import CrowdCounterModel

# Flask app setup
app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": ["http://77.38.76.152", "http://localhost"]}})

# Configure upload folder
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Define transformation with smaller image size
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

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
        # Select and load the required model dynamically
        if density_type == 'dense':
            model = CrowdCounterModel().cuda()
            checkpoint = torch.load("1model_best.pth.tar", map_location="cuda", weights_only=True)
            model.load_state_dict(checkpoint['state_dict'])
        elif density_type == 'sparse':
            model = CrowdCounterModel().cuda()
            checkpoint = torch.load("2model_best.pth.tar", map_location="cuda", weights_only=True)
            model.load_state_dict(checkpoint['state_dict'])
        else:
            return jsonify({'error': 'Invalid density type. Choose "dense" or "sparse".'}), 400

        model.eval()

        # Process image
        img = transform(Image.open(filepath).convert('RGB')).cuda()
        with torch.no_grad():
            output = model(img.unsqueeze(0))
            predicted_count = int(output.detach().cpu().sum().item())

        # Clean up uploaded file
        os.remove(filepath)

        # Unload model from GPU memory
        del model
        torch.cuda.empty_cache()

        return jsonify({'predicted_count': predicted_count})

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    # Optionally set CUDA environment variable for memory management
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
    app.run(host='0.0.0.0', port=5000, debug=True)
