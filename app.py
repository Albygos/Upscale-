import os
import math
import io
import gc
import torch
import torch.nn as nn
from flask import Flask, request, send_file
from torchvision import transforms
from PIL import Image

app = Flask(__name__)

# ==========================================
# 1. MODEL ARCHITECTURE
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
# 2. MODEL INITIALIZATION (CPU ONLY)
# ==========================================
DEVICE = torch.device("cpu")
model = SRResNet(upscale_factor=4).to(DEVICE)
model_path = "4k_dslr_clarity_model.pth"

if os.path.exists(model_path):
    model.load_state_dict(torch.load(model_path, map_location=DEVICE, weights_only=True))
    print("Model loaded successfully.")
else:
    print(f"Warning: '{model_path}' not found. Inference will run with random weights.")

model.eval()

# ==========================================
# 3. HTML INTERFACE & ROUTES
# ==========================================
HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>4K AI Upscaler - Render</title>
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <style>
        body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; max-width: 720px; margin: 40px auto; padding: 0 16px; text-align: center; background: #f9f9f9; }
        .card { background: white; border: 1px solid #e0e0e0; padding: 32px; border-radius: 12px; box-shadow: 0 4px 6px rgba(0,0,0,0.05); }
        input[type="file"] { margin: 20px 0; }
        button { padding: 12px 24px; background: #4f46e5; color: white; border: none; border-radius: 6px; font-weight: bold; cursor: pointer; }
        button:disabled { background: #9ca3af; cursor: not-allowed; }
        #status { margin-top: 16px; color: #4b5563; font-size: 14px; }
        img { max-width: 100%; margin-top: 24px; border-radius: 8px; border: 1px solid #ddd; }
    </style>
</head>
<body>
    <div class="card">
        <h2>4K AI Image Upscaler</h2>
        <p>Upload a low-resolution image to upscale it 4x using SRResNet.</p>
        <form id="uploadForm">
            <input type="file" id="imageInput" name="image" accept="image/*" required><br>
            <button type="submit" id="submitBtn">Enhance Resolution</button>
        </form>
        <div id="status"></div>
        <img id="resultImage" style="display:none;" />
    </div>

    <script>
        document.getElementById('uploadForm').onsubmit = async (e) => {
            e.preventDefault();
            const form = e.target;
            const status = document.getElementById('status');
            const resultImg = document.getElementById('resultImage');
            const btn = document.getElementById('submitBtn');

            btn.disabled = true;
            status.textContent = "Processing... Render's free tier is slow, this may take up to 60 seconds.";
            resultImg.style.display = 'none';

            try {
                const response = await fetch('/upscale', {
                    method: 'POST',
                    body: new FormData(form)
                });
                if (!response.ok) throw new Error(await response.text());

                const blob = await response.blob();
                resultImg.src = URL.createObjectURL(blob);
                resultImg.style.display = 'block';
                status.textContent = "Upscale complete!";
            } catch (err) {
                status.textContent = "Error: " + err.message;
            } finally {
                btn.disabled = false;
            }
        };
    </script>
</body>
</html>
"""

@app.route('/')
def home():
    return HTML_TEMPLATE

@app.route('/upscale', methods=['POST'])
def upscale():
    if 'image' not in request.files:
        return "No image uploaded", 400
        
    file = request.files['image']
    if file.filename == '':
        return "No selected file", 400

    try:
        img = Image.open(file.stream).convert('RGB')
        
        # EXTREME RAM SAFEGUARD FOR RENDER FREE TIER (512MB RAM Limit)
        # Cap to 250px. 250px inputs scale up to 1000px outputs perfectly.
        # Anything larger will crash the OOM Killer.
        max_dim = 250
        if img.width > max_dim or img.height > max_dim:
            img.thumbnail((max_dim, max_dim), Image.Resampling.LANCZOS)
            
        transform = transforms.ToTensor()
        input_tensor = transform(img).unsqueeze(0).to(DEVICE)
        
        # Use inference_mode instead of no_grad (Uses less memory and runs faster)
        with torch.inference_mode():
            output_tensor = model(input_tensor).squeeze(0)
            
        to_pil = transforms.ToPILImage()
        output_image = to_pil(output_tensor)
        
        # Force garbage collection immediately to dump PyTorch's temporary calculations
        del input_tensor, output_tensor
        gc.collect()
        
        img_io = io.BytesIO()
        output_image.save(img_io, 'JPEG', quality=95)
        img_io.seek(0)
        
        return send_file(img_io, mimetype='image/jpeg')
        
    except Exception as e:
        return str(e), 500

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
