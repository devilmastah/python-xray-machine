#import numpy as np
#import tifffile
#import matplotlib.pyplot as plt
#from skimage.restoration import richardson_lucy
#
## Load 16-bit grayscale TIFF image
#image = tifffile.imread("median_stack_29-01-2025 10-52.tiff").astype(np.float32)
#image /= np.max(image)  # Normalize to [0,1]
#
## Define PSF (Gaussian approximation)
#psf_size = 5
#psf = np.outer(np.hanning(psf_size), np.hanning(psf_size))
#psf /= psf.sum()
#
## Apply Richardson-Lucy Deconvolution
#deconvolved = richardson_lucy(image, psf, num_iter=50)
#
## Convert back to 16-bit
#deconvolved_16bit = (deconvolved * 65535).clip(0, 65535).astype(np.uint16)
#
## Save result
#tifffile.imwrite("deblurred_xray_RichardsonLucy.tiff", deconvolved_16bit)
#
## Show Images
#fig, ax = plt.subplots(1, 2, figsize=(10, 5))
#ax[0].imshow(image, cmap="gray")
#ax[0].set_title("Original Image")
#
#ax[1].imshow(deconvolved, cmap="gray")
#ax[1].set_title("Richardson-Lucy Deconvolved Image")
#
#plt.show()




import numpy as np
import cv2
import tifffile
import matplotlib.pyplot as plt
from skimage.restoration import richardson_lucy

# ✅ Load 16-bit grayscale TIFF image
image = tifffile.imread("median_stack_03-02-2025 17-42_DC.tiff").astype(np.float32)

# ✅ Normalize to 0-1 range (keeping full precision)
image /= 65535.0  

# ==========================
# ✅ Apply CLAHE Directly in 16-bit
# ==========================
def apply_clahe_16bit(image, clip_limit=1.0, tile_grid_size=(8, 8)):
    """Applies CLAHE while preserving 16-bit dynamic range."""
    # Convert to 16-bit integer
    image_16bit = (image * 65535).astype(np.uint16)

    # Create CLAHE object
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)

    # Apply CLAHE
    enhanced_16bit = clahe.apply(image_16bit)

    # Convert back to float while preserving 16-bit range
    return enhanced_16bit.astype(np.float32) / 65535.0

# ✅ Apply CLAHE
clahe_image = apply_clahe_16bit(image)

# ✅ Define PSF (Gaussian Blur) for Deconvolution
psf_size = 3  # Keep odd-sized
psf_sigma = 1  
x = np.linspace(-psf_size // 2, psf_size // 2, psf_size)
y = np.linspace(-psf_size // 2, psf_size // 2, psf_size)
X, Y = np.meshgrid(x, y)
psf = np.exp(-(X**2 + Y**2) / (2 * psf_sigma**2))
psf /= psf.sum()  # Normalize

# ==========================
# ✅ 1️⃣ OpenCV Wiener Deconvolution (16-bit)
# ==========================
def wiener_deconvolution_opencv(image, psf, K=0.005):
    """Applies Wiener deconvolution using OpenCV in 16-bit mode."""
    image_fft = np.fft.fft2(image)
    psf_fft = np.fft.fft2(psf, s=image.shape)
    psf_fft_conj = np.conj(psf_fft)

    # Wiener filtering formula
    deconvolved_fft = (image_fft * psf_fft_conj) / (psf_fft * psf_fft_conj + K)

    # Convert back to spatial domain
    deconvolved = np.abs(np.fft.ifft2(deconvolved_fft))

    # Preserve intensity scaling
    return deconvolved

# ✅ Apply Wiener Deconvolution
wiener_deconvolved = wiener_deconvolution_opencv(image, psf)
wiener_deconvolved_clahe = wiener_deconvolution_opencv(clahe_image, psf)

# ==========================
# ✅ 2️⃣ Lucy-Richardson Deconvolution (16-bit)
# ==========================
lucy_deconvolved = richardson_lucy(image, psf, num_iter=30)
lucy_deconvolved_clahe = richardson_lucy(clahe_image, psf, num_iter=30)

# ==========================
# ✅ 3️⃣ Van Cittert Deconvolution (16-bit)
# ==========================
def van_cittert_deconvolution(image, psf, iterations=30):
    """Applies Van Cittert deconvolution for 16-bit images."""
    estimate = image.copy()
    for _ in range(iterations):
        estimate += cv2.filter2D(image - cv2.filter2D(estimate, -1, psf), -1, psf)
    return estimate

# ✅ Apply Van Cittert Deconvolution
van_cittert_deconvolved = van_cittert_deconvolution(image, psf, iterations=30)
van_cittert_deconvolved_clahe = van_cittert_deconvolution(clahe_image, psf, iterations=30)

# ==========================
# ✅ Convert & Save Results in Full 16-bit TIFF Format
# ==========================
def save_tiff(image, filename):
    """Saves image in 16-bit TIFF format while preserving full dynamic range."""
    tiff_image = (image * 65535).clip(0, 65535).astype(np.uint16)
    tifffile.imwrite(filename, tiff_image)

save_tiff(clahe_image, "Clahe.tiff")
# ✅ Save Deconvolutions with CLAHE
save_tiff(wiener_deconvolved_clahe, "deblurred_xray_Wiener_CLAHE.tiff")
save_tiff(lucy_deconvolved_clahe, "deblurred_xray_LucyRichardson_CLAHE.tiff")
save_tiff(van_cittert_deconvolved_clahe, "deblurred_xray_VanCittert_CLAHE.tiff")

# ✅ Save Deconvolutions Without CLAHE
save_tiff(wiener_deconvolved, "deblurred_xray_Wiener.tiff")
save_tiff(lucy_deconvolved, "deblurred_xray_LucyRichardson.tiff")
save_tiff(van_cittert_deconvolved, "deblurred_xray_VanCittert.tiff")

# ==========================
# ✅ Display Results
# ==========================
fig, ax = plt.subplots(2, 4, figsize=(15, 10))

ax[0, 0].imshow(image, cmap="gray")
ax[0, 0].set_title("Original Image")

ax[0, 1].imshow(wiener_deconvolved, cmap="gray")
ax[0, 1].set_title("Wiener Deconvolution")

ax[0, 2].imshow(lucy_deconvolved, cmap="gray")
ax[0, 2].set_title("Lucy-Richardson Deconvolution")

ax[0, 3].imshow(van_cittert_deconvolved, cmap="gray")
ax[0, 3].set_title("Van Cittert Deconvolution")

ax[1, 0].imshow(clahe_image, cmap="gray")
ax[1, 0].set_title("CLAHE Image")

ax[1, 1].imshow(wiener_deconvolved_clahe, cmap="gray")
ax[1, 1].set_title("Wiener Deconvolution (CLAHE)")

ax[1, 2].imshow(lucy_deconvolved_clahe, cmap="gray")
ax[1, 2].set_title("Lucy-Richardson (CLAHE)")

ax[1, 3].imshow(van_cittert_deconvolved_clahe, cmap="gray")
ax[1, 3].set_title("Van Cittert (CLAHE)")

plt.tight_layout()
plt.show()

print("✅ Deblurring complete! Saved results as:")
print("- deblurred_xray_Wiener.tiff")
print("- deblurred_xray_LucyRichardson.tiff")
print("- deblurred_xray_VanCittert.tiff")
print("- deblurred_xray_Wiener_CLAHE.tiff")
print("- deblurred_xray_LucyRichardson_CLAHE.tiff")
print("- deblurred_xray_VanCittert_CLAHE.tiff")






#import numpy as np
#import cv2
#import tifffile
#import matplotlib.pyplot as plt
#from scipy.optimize import curve_fit
#from scipy.fftpack import fft2, fftshift, ifft2
#from scipy.ndimage import gaussian_filter1d, median_filter
#
## ✅ Load 16-bit grayscale TIFF image
#image = tifffile.imread("median_stack_29-01-2025 10-52.tiff").astype(np.float32)
#image /= 65535.0  # Normalize
#
## ==========================
## ✅ Method 1: Edge Spread Function (ESF)
## ==========================
#def estimate_psf_esf(image):
#    """Estimate PSF using Edge Spread Function (Gaussian Fit)."""
#    edges = cv2.Sobel(image, cv2.CV_64F, 1, 1, ksize=5)
#    edges = np.abs(edges)
#    edge_profile = np.sum(edges, axis=1)
#    edge_index = np.argmax(edge_profile)
#
#    # Extract region around the strongest edge
#    crop_size = 50
#    start_idx = max(edge_index - crop_size // 2, 0)
#    end_idx = min(edge_index + crop_size // 2, image.shape[0])
#
#    # Extract a line profile
#    line_profile = np.mean(image[start_idx:end_idx, :], axis=0)
#    line_profile = gaussian_filter1d(line_profile, sigma=2)
#    line_profile = median_filter(line_profile, size=3)
#
#    # Define Gaussian function
#    def gaussian(x, A, mu, sigma, C):
#        return A * np.exp(-(x - mu) ** 2 / (2 * sigma ** 2)) + C
#
#    # Fit Gaussian
#    x_data = np.arange(len(line_profile))
#    A_guess = np.max(line_profile) - np.min(line_profile)
#    mu_guess = np.argmax(line_profile)
#    sigma_guess = 10
#    C_guess = np.min(line_profile)
#
#    try:
#        popt, _ = curve_fit(gaussian, x_data, line_profile, p0=[A_guess, mu_guess, sigma_guess, C_guess], maxfev=5000)
#        psf_sigma_esf = popt[2]
#        psf_size_esf = int(2 * psf_sigma_esf)
#
#        print(f"✅ [ESF] Estimated PSF Sigma: {psf_sigma_esf:.2f}")
#        print(f"✅ [ESF] Suggested PSF Size: {psf_size_esf}")
#
#        return psf_sigma_esf, psf_size_esf
#    except RuntimeError:
#        print("❌ [ESF] Gaussian Fit Failed.")
#        return None, None
#
## ==========================
## ✅ Method 2: Fast Autocorrelation Analysis (Optimized)
## ==========================
#def estimate_psf_autocorr_fast(image):
#    """Estimate PSF using fast autocorrelation (Fourier Transform)."""
#    image_fft = fft2(image)
#    autocorr = np.abs(ifft2(image_fft * np.conj(image_fft)))
#
#    # Find the blur radius from the central peak width
#    center_x, center_y = np.array(autocorr.shape) // 2
#    cross_section = autocorr[center_x, :]  # Take 1D slice across the center
#    psf_size_autocorr = np.argmax(cross_section < 0.5 * np.max(cross_section))  # Half-max width
#
#    print(f"✅ [Fast Autocorrelation] Estimated PSF Size: {psf_size_autocorr}")
#    return psf_size_autocorr
#
## ==========================
## ✅ Method 3: Frequency Domain Analysis
## ==========================
#def estimate_psf_frequency(image):
#    """Estimate PSF using frequency domain analysis."""
#    image_fft = np.abs(fftshift(fft2(image)))  # Compute power spectrum
#
#    # Compute Radial Power Spectrum (for defocus blur estimation)
#    y, x = np.indices(image.shape)
#    center = np.array(image.shape) // 2
#    r = np.sqrt((x - center[1]) ** 2 + (y - center[0]) ** 2)
#    radial_profile = np.bincount(r.ravel().astype(int), weights=image_fft.ravel())
#    psf_size_freq = np.argmax(radial_profile < 0.5 * np.max(radial_profile))  # Half-max width
#
#    # Display the power spectrum
#    plt.figure(figsize=(5, 5))
#    plt.imshow(np.log1p(image_fft), cmap="inferno")
#    plt.title("Power Spectrum of Image")
#    plt.show()
#
#    print(f"✅ [Frequency] Estimated PSF Size: {psf_size_freq}")
#    return psf_size_freq
#
## ==========================
## ✅ Run All PSF Estimation Methods
## ==========================
#psf_sigma_esf, psf_size_esf = estimate_psf_esf(image)
#psf_size_autocorr = estimate_psf_autocorr_fast(image)  # ✅ Now optimized!
#psf_size_freq = estimate_psf_frequency(image)
#
#print("\n🎯 Final PSF Estimates:")
#print(f"🔹 ESF Method (Gaussian Fit) -> Sigma: {psf_sigma_esf:.2f}, Size: {psf_size_esf}")
#print(f"🔹 Autocorrelation Method -> Size: {psf_size_autocorr}")
#print(f"🔹 Frequency Domain Method -> Size: {psf_size_freq}")
#