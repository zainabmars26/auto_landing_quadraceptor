import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from transformers import SegformerForSemanticSegmentation, SegformerImageProcessor

# 1. Setup device and hardware configurations
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 2. Define our 8 classes and palette
UAVID_PALETTE = np.array([
    [0, 0, 0],         # 0: Clutter
    [128, 0, 0],       # 1: Building
    [128, 64, 128],    # 2: Road
    [0, 128, 0],       # 3: Tree
    [128, 128, 0],     # 4: Vegetation
    [64, 0, 128],      # 5: Moving Car
    [192, 0, 192],     # 6: Static Car
    [64, 64, 0]        # 7: Human
], dtype=np.uint8)

id2label = {0: "Clutter", 1: "Building", 2: "Road", 3: "Tree", 4: "Vegetation", 5: "Moving_Car", 6: "Static_Car", 7: "Human"}
label2id = {v: k for k, v in id2label.items()}

# 3. Load the model architecture and custom trained weights
processor = SegformerImageProcessor.from_pretrained("nvidia/mit-b0")
model = SegformerForSemanticSegmentation.from_pretrained(
    "nvidia/mit-b0", num_labels=8, id2label=id2label, label2id=label2id, ignore_mismatched_sizes=True
)
model.load_state_dict(torch.load(r"segformer_uavid_epoch_77-9-7.pt", map_location=device)) #add pretrained model 
model.to(device)
model.eval()

# 4. Load a test image from your validation set
# (Change this path to point to a real image file in your val folder)
test_image_path = r"C:\Users\Zainab.Alawneh\Desktop\tasks\AirSim\dataset\New folder\manual_snap_1783425546.png"
raw_image = Image.open(test_image_path).convert("RGB")
original_size = raw_image.size  # Keep track of original dimensions (Width, Height)

# Pre-process image to 512x512 tensor
inputs = processor(images=raw_image, return_tensors="pt")
pixel_values = inputs["pixel_values"].to(device)

# 5. Forward Pass (Inference)
with torch.no_grad():
    outputs = model(pixel_values=pixel_values)
    logits = outputs.logits  # Shape: [1, 8, 128, 128] (SegFormer outputs are 1/4 of input size)

# 6. Resize logits back to the original image dimensions
# This performs bilinear upsampling to give us a crisp mask matching our photo
upsampled_logits = torch.nn.functional.interpolate(
    logits, size=original_size[::-1], mode="bilinear", align_corners=False
)

# Get the single highest probability class index for each pixel
prediction_indices = upsampled_logits.argmax(dim=1).squeeze(0).cpu().numpy()

# 7. Map the 2D indices (0-7) back to beautiful RGB colors
color_mask = UAVID_PALETTE[prediction_indices]

# 8. Plot the Results side-by-side
fig, ax = plt.subplots(1, 2, figsize=(12, 6))
ax[0].imshow(raw_image)
ax[0].set_title("Original Drone Input Frame")
ax[0].axis("off")

ax[1].imshow(color_mask)
ax[1].set_title("SegFormer 8-Class Prediction Map")
ax[1].axis("off")

plt.tight_layout()
plt.show()