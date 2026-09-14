import cv2
import numpy as np

def aggressive_city_sample_look(image_path):
    # 1. Load image
    img = cv2.imread(image_path)
    if img is None:
        return None
    
    # --- STEP 1: FORCE DEEP CONTRAST & CRUSH SHADOWS ---
    # Convert to float for precise curve operations
    f_img = img.astype(np.float32) / 255.0
    
    # Apply a high-contrast S-curve to punch up shadows and highlights
    # Formula: 3*x^2 - 2*x^3
    s_curve = 3 * (f_img ** 2) - 2 * (f_img ** 3)
    
    # Clip and scale back to 8-bit
    img_contrast = np.clip(s_curve * 255.0, 0, 255).astype(np.uint8)

    # --- STEP 2: AGGRESIVE COLOR GRADING (THE MATRIX PALETTE) ---
    # Shift to LAB to isolate color from luminance
    lab = cv2.cvtColor(img_contrast, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    
    # Mute the greens/reds significantly
    a = cv2.addWeighted(a, 0.65, np.full_like(a, 128), 0.35, 0)
    
    # Pull the B channel heavily away from yellow and deep into cool blues/cyans
    b = cv2.addWeighted(b, 0.60, np.full_like(b, 128), 0.40, 0)
    b = cv2.subtract(b, 12)  # Much stronger cyan-blue shift
    
    graded = cv2.merge([l, a, b])
    img_graded = cv2.cvtColor(graded, cv2.COLOR_LAB2BGR)

    # --- STEP 3: SIMULATE UE5 LUMEN / AMBIENT OCCLUSION ---
    # We create a dark mask based on the intensity of the image to artificially
    # deepen crevices, asphalt, and building edges.
    gray = cv2.cvtColor(img_graded, cv2.COLOR_BGR2GRAY)
    shadow_mask = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 15, 7
    )
    shadow_mask_blur = cv2.GaussianBlur(shadow_mask, (5, 5), 0)
    
    # Blend the dark micro-shadows back in
    shadow_factor = shadow_mask_blur.astype(np.float32) / 255.0
    img_ao = img_graded.astype(np.float32)
    for i in range(3):
        img_ao[:, :, i] = img_ao[:, :, i] * (0.8 + 0.2 * shadow_factor)
    img_ao = np.clip(img_ao, 0, 255).astype(np.uint8)

    # --- STEP 4: HEAVY CINEMATIC LENS FALLOFF ---
    rows, cols = img_ao.shape[:2]
    kernel_x = cv2.getGaussianKernel(cols, cols * 0.4)
    kernel_y = cv2.getGaussianKernel(rows, rows * 0.4)
    kernel = kernel_y * kernel_x.T
    mask = kernel / kernel.max()
    
    # Make the vignette stronger at the corners
    vignette = np.copy(img_ao)
    for i in range(3):
        vignette[:, :, i] = vignette[:, :, i] * mask
        
    final_out = cv2.addWeighted(img_ao, 0.3, vignette, 0.7, 0)
    
    return final_out
# Example usage:
processed_img = aggressive_city_sample_look('000100.png')
cv2.imwrite('uavid_city_sample_style.png', processed_img)