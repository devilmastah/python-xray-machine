import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2
import numpy as np
import os

# Global variables to store device and model references after initialization
device = None
model = None

# ========================
# 1. Model Architecture (Same as Training)
# ========================
class ResidualBlock(nn.Module):
    def __init__(self, in_channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, padding_mode='reflect')
        self.relu = nn.LeakyReLU(negative_slope=0.1, inplace=True)
        self.conv2 = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, padding_mode='reflect')

    def forward(self, x):
        return x + self.conv2(self.relu(self.conv1(x)))

class UNetResidual(nn.Module):
    def __init__(self):
        super(UNetResidual, self).__init__()
        # Encoder
        self.enc1 = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, padding=1),
            nn.LeakyReLU(0.1, inplace=True),
            ResidualBlock(64)
        )
        self.enc2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(0.1, inplace=True),
            ResidualBlock(128)
        )
        self.enc3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(0.1, inplace=True),
            ResidualBlock(256)
        )
        # Decoder
        self.dec2 = nn.Sequential(
            ResidualBlock(256),
            nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2),
            nn.LeakyReLU(0.1, inplace=True)
        )
        self.dec1 = nn.Sequential(
            ResidualBlock(128),
            nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2),
            nn.LeakyReLU(0.1, inplace=True)
        )
        # Final output
        self.final = nn.Conv2d(64, 1, kernel_size=3, padding=1)

    def forward(self, x):
        x1 = self.enc1(x)
        x2 = self.enc2(x1)
        x3 = self.enc3(x2)

        x4 = self.dec2(x3) + x2  # Skip connection
        x5 = self.dec1(x4) + x1  # Skip connection
        return self.final(x5)

# ========================
# 2. Initialization Function
# ========================
def init_model(model_path="best_deconvolution_model.pth", use_cuda_if_available=True):
    """
    Initializes the device and loads the trained model from disk.
    Call this once at the start of your script.
    """
    global device, model

    if use_cuda_if_available and torch.cuda.is_available():
        device = torch.device("cuda:0")
    else:
        device = torch.device("cpu")

    print(f"✅ Initializing model on device: {device}")

    model = UNetResidual().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    print(f"✅ Model loaded from {model_path} and set to eval mode.")


# ========================
# 3. Multiple-of-4 Padding Helper
# ========================
def _pad_to_multiple_of(value, multiple=4):
    """
    Returns how many pixels we need to pad so that 'value' becomes a multiple of 'multiple'.
    Example: if value=901 and multiple=4, remainder=1, so we pad=3 to get 904.
    """
    remainder = value % multiple
    if remainder == 0:
        return 0
    else:
        return multiple - remainder

def _preprocess_image(image_data: np.ndarray):
    """
    1) Convert 16-bit to float [0,1] if needed
    2) Pad each dimension to nearest multiple of 4
    3) Return the padded image as float32, original shape, and padding info
    """
    # Convert if 16-bit
    if image_data.dtype == np.uint16:
        image_data = image_data.astype(np.float32) / 65535.0
        image_data = image_data * 0.85


    orig_H, orig_W = image_data.shape

    # Calculate how many pixels to pad for each dimension
    pad_H = _pad_to_multiple_of(orig_H, 4)
    pad_W = _pad_to_multiple_of(orig_W, 4)

    if pad_H > 0 or pad_W > 0:
        image_data = np.pad(image_data, ((0, pad_H), (0, pad_W)), mode='reflect')

    return image_data, (orig_H, orig_W), (pad_H, pad_W)

# ========================
# 4. deconvoluteFrame: In-Memory Deconvolution
# ========================
def deconvoluteFrame(image_data: np.ndarray, autocast_inference=True) -> np.ndarray:
    """
    Deconvolves a single 2D grayscale image array in memory.
    Expects image_data shape: [H, W], 16-bit or float in [0,1].
    Returns a 16-bit result (np.uint16) with the same original size.
    """
    global device, model
    if model is None:
        raise RuntimeError("Model not initialized. Call init_model() first.")

    # 1. Preprocess (convert/pad)
    img_padded, (orig_H, orig_W), _ = _preprocess_image(image_data)

    # 2. Convert to tensor
    input_tensor = torch.from_numpy(img_padded).unsqueeze(0).unsqueeze(0).to(device)

    # 3. Inference
    with torch.no_grad():
        if autocast_inference and device.type == 'cuda':
            with torch.amp.autocast(device_type='cuda'):
                output_tensor = model(input_tensor)
        else:
            output_tensor = model(input_tensor)

    # 4. Convert back to NumPy, scale to 16-bit, crop
    output_image = output_tensor.squeeze().cpu().numpy()
    # 1. Scale the float values by 1.15
    output_image *= 1.15
    # 2. Clip them in [0,1] (if you prefer to keep them in float range before scaling to 16-bit)
    output_image = np.clip(output_image, 0.0, 1.0)
    
    output_image = (output_image * 65535.0).astype(np.uint16)
    output_image = output_image[:orig_H, :orig_W]

   # # 5. Post-Process:
   # #    Swap pure black (0) with full white (65535)
   # zero_mask = (output_image < 2000)
   # full_mask = (output_image > 63535)
   # output_image[zero_mask] = 63535
   # output_image[full_mask] = 2000

    return output_image

# ========================
# 5. deconvoluteFile: File I/O Deconvolution
# ========================
def deconvoluteFile(input_path: str, output_path: str, autocast_inference=True):
    """
    Loads a TIFF from 'input_path', deconvolutes it, and writes the result as 16-bit TIFF to 'output_path'.
    """
    # Load image in 16-bit or float
    image_data = cv2.imread(input_path, cv2.IMREAD_UNCHANGED)
    if image_data is None:
        raise FileNotFoundError(f"Could not load image: {input_path}")

    # Deconvolute using the in-memory function
    result_image = deconvoluteFrame(image_data, autocast_inference=autocast_inference)

    # Save output
    cv2.imwrite(output_path, result_image)
    print(f"✅ Deconvolution saved to {output_path}")
