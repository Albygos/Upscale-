import os
import math
import io
import torch
import torch.nn as nn
from flask import Flask, request, send_file
from torchvision import transforms
from PIL import Image

app = Flask(__name__)

# ==========================================
# 1. AI ARCHITECTURE
# ==========================================
class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.prelu = nn.PReLU()
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        return x + self.bn2(self.conv2(self.prelu(self.bn1(self.conv1(x)))))

class SRResNet(nn.Module):
    def __init__(self, upscale_factor=4, num_residual_blocks=16):
        super(SRResNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=9, padding=4)
        self.prelu = nn.PReLU()
        self.res_blocks = nn.Sequential(*[ResidualBlock(64) for _ in range(num_residual_blocks)])
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        
        upsample_blocks = []
        for _ in range(int(math.log2(upscale_factor))):
            upsample_blocks.append(nn.Conv2d(64, 64 * 4, kernel_size=3, padding=1))
            upsample_blocks.append(nn.PixelShuffle(2))
            upsample_blocks.append(nn.PReLU())
        self.upsample_blocks = nn.Sequential(*upsample_blocks)
        self.conv3 = nn.Conv2d(64, 3, kernel_size=9, padding=4)

    def forward(self, x):
        x1 = self.prelu(self.conv1(x))
        x2 = self.bn2(self.conv2(self.res_blocks(x1)))
        return torch.sigmoid(self.conv3(self.upsample_blocks(x1 + x2)))

# ==========================================
# 2. INITIALIZE CPU MODEL
# ==========================================
# Vercel does not have GPUs. We must force CPU mapping.
DEVICE = torch.device("cpu")
model = SRResNet(upscale_factor=4).to(DEVICE)
model_path = "4k_dslr_clarity_model.pth"

if os.path.exists(model_path):
    model.load_state_dict(torch.load(model_path, map_location=DEVICE, weights_only=True))
model.eval()

# ==========================================
# 3. FLASK ROUTES
# ==========================================
HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>4K AI Upscaler</title>
    <style>
        body { font-family: sans-serif; max-width: 800px; margin: 40px auto; text-align: center; }
        .container { border: 2px dashed #ccc; padding: 40px; border-radius: 10px; }
        img { max-width: 100%; margin-top: 20px; border-radius: 8px; }
        button { padding: 10px 20px; background: #0070f3; color: white; border: none; border-radius: 5px; cursor: pointer; }
    </style>
</head>
<body>
    <div class="container">
        <h2>Upload Low-Res Photo</h2>
        <form id="uploadForm" enctype="multipart/form-data">
            <input type="file" id="imageInput" name="image" accept="image/*" required>
            <br><br>
            <button type="submit" id="submitBtn">Upscale (Max 300px)</button>
        </form>
        <div id="loading" style="display:none; margin-top: 20px;">Processing on CPU... please wait up to 10 seconds.</div>
        <img id="resultImage" style="display:none;" />
    </div>

    <script>
        document.getElementById('uploadForm').onsubmit = async (e) => {
            e.preventDefault();
            const formData = new FormData(e.target);
            document.getElementById('loading').style.display = 'block';
            document.getElementById('resultImage').style.display = 'none';
            document.getElementById('submitBtn').disabled = true;

            try {
                const response = await fetch('/upscale', { method: 'POST', body: formData });
                if (!response.ok) throw new Error(await response.text());
                
                const blob = await response.blob();
                document.getElementById('resultImage').src = URL.createObjectURL(blob);
                document.getElementById('resultImage').style.display = 'block';
            } catch (error) {
                alert('Error: ' + error.message);
            } finally {
                document.getElementById('loading').style.display = 'none';
                document.getElementById('submitBtn').disabled = false;
            }
        };
    </script>
</body>
</html>
"""

@app.route('/')
def index():
    return HTML_TEMPLATE

@app.route('/upscale', methods=['POST'])
def upscale():
    if 'image' not in request.files:
        return "No image uploaded", 400
        
    file = request.files['image']
    if file.filename == '':
        return "No selected file", 400

    try:
        # Load and verify image
        img = Image.open(file.stream).convert('RGB')
        
        # Vercel CPU restriction: large images will exceed the 10-second timeout
        max_dim = 300
        if img.width > max_dim or img.height > max_dim:
            img.thumbnail((max_dim, max_dim), Image.Resampling.LANCZOS)
            
        # Transform and infer
        transform = transforms.ToTensor()
        input_tensor = transform(img).unsqueeze(0).to(DEVICE)
        
        with torch.no_grad():
            output_tensor = model(input_tensor).squeeze(0)
            
        # Convert back to image in memory
        to_pil = transforms.ToPILImage()
        output_image = to_pil(output_tensor)
        
        # Save to BytesIO object (Vercel filesystem is read-only)
        img_io = io.BytesIO()
        output_image.save(img_io, 'JPEG', quality=95)
        img_io.seek(0)
        
        return send_file(img_io, mimetype='image/jpeg')
        
    except Exception as e:
        return str(e), 500

if __name__ == '__main__':
    app.run(debug=True)
