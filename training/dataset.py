import os
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
from transformers import SegformerImageProcessor

# The exact 8 RGB colors used in the UAVid dataset masks
UAVID_PALETTE = [
    [0, 0, 0],         # 0: Clutter
    [128, 0, 0],       # 1: Building
    [128, 64, 128],    # 2: Road
    [0, 128, 0],       # 3: Tree
    [128, 128, 0],     # 4: Vegetation
    [64, 0, 128],      # 5: Moving Car
    [192, 0, 192],     # 6: Static Car
    [64, 64, 0]        # 7: Human
]

class UAVidDataset(Dataset):
    def __init__(self, root_dir, image_processor, target_size=(512, 512)):
        """
        root_dir: Path to 'train' or 'val' folder (e.g., 'uavid_dataset/train')
        """
        self.img_dir = os.path.join(root_dir, "images")
        self.mask_dir = os.path.join(root_dir, "masks")
        self.img_names = sorted(os.listdir(self.img_dir))
        self.processor = image_processor
        self.target_size = target_size

    def __len__(self):
        return len(self.img_names)

    def rgb_to_index_mask(self, rgb_mask):
        """Converts (H, W, 3) RGB mask to (H, W) index mask (0-7)"""
        mask_array = np.array(rgb_mask)
        index_mask = np.zeros((mask_array.shape[0], mask_array.shape[1]), dtype=np.uint8)
        
        for class_id, color in enumerate(UAVID_PALETTE):
            match = np.all(mask_array == color, axis=-1)
            index_mask[match] = class_id
            
        return index_mask

    def __getitem__(self, idx):
        # 1. Load image and corresponding mask
        img_path = os.path.join(self.img_dir, self.img_names[idx])
        mask_path = os.path.join(self.mask_dir, self.img_names[idx]) # Assumes identical names
        
        image = Image.open(img_path).convert("RGB")
        rgb_mask = Image.open(mask_path).convert("RGB")
        
        # 2. Resize both to ensure they match perfectly
        image = image.resize(self.target_size, Image.BILINEAR)
        rgb_mask = rgb_mask.resize(self.target_size, Image.NEAREST) # NEAREST prevents color bleeding!
        
        # 3. Convert mask from RGB colors to integers 0-7
        index_mask = self.rgb_to_index_mask(rgb_mask)
        
        # 4. Use SegFormer processor to scale pixel values correctly
        encoded_inputs = self.processor(images=image, return_tensors="pt")
        
        # Remove batch dimension added by processor default behavior
        inputs = {k: v.squeeze(0) for k, v in encoded_inputs.items()}
        
        # Add our processed target mask tensor
        inputs["labels"] = torch.tensor(index_mask, dtype=torch.long)
        
        return inputs



from torch.utils.data import DataLoader

# Initialize the image processor from Hugging Face
processor = SegformerImageProcessor.from_pretrained("nvidia/mit-b0")

# Create your training dataset instance
# (Replace 'uavid_dataset/train' with the actual path on your computer)
train_dataset = UAVidDataset(root_dir="training data/train", image_processor=processor)

# Load a single sample batch
train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)
first_batch = next(iter(train_loader))

print("--- Data Pipeline Check ---")
print("Pixel values shape (Images):", first_batch["pixel_values"].shape) 
print("Labels shape (Ground Truth Masks):", first_batch["labels"].shape)