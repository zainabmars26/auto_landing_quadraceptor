import torch
from torch.utils.data import DataLoader
from transformers import SegformerForSemanticSegmentation, SegformerImageProcessor
from dataset import UAVidDataset  # Assumes your dataset class is in dataset.py
from tqdm import tqdm

# 1. Hardware Configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# 2. Initialize Processor and Datasets
processor = SegformerImageProcessor.from_pretrained("nvidia/mit-b0")

# Adjust paths to match your local setup
train_dataset = UAVidDataset(root_dir="augmented_dataset/train", image_processor=processor)
val_dataset = UAVidDataset(root_dir="augmented_dataset/val", image_processor=processor)

train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True) # Increased batch size for training
val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False)

# 3. Initialize SegFormer with 8 Classes
id2label = {0: "Clutter", 1: "Building", 2: "Road", 3: "Tree", 4: "Vegetation", 5: "Moving_Car", 6: "Static_Car", 7: "Human"}
label2id = {v: k for k, v in id2label.items()}

model = SegformerForSemanticSegmentation.from_pretrained(
    "nvidia/mit-b0",
    num_labels=8,
    id2label=id2label,
    label2id=label2id,
    ignore_mismatched_sizes=True
)
checkpoint_path = r"C:\Users\Zainab.Alawneh\Desktop\Traversability Estimation_seg_former\segformer_uavid_epoch_200_.pt"
model.load_state_dict(torch.load(checkpoint_path, map_location=device))
print(f"Loaded best checkpoint: {checkpoint_path} as the starting point for new training.")

model.to(device)

# 4. Optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=6e-5)

# 5. Training Loop
EPOCHS = 200  # Start small to make sure everything runs

for epoch in range(EPOCHS):
    model.train()
    epoch_loss = 0
    
    # Progress bar wrapper
    progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}")
    
    for batch in progress_bar:
        optimizer.zero_grad()
        
        # Move inputs to GPU/CPU
        pixel_values = batch["pixel_values"].to(device)
        labels = batch["labels"].to(device)
        
        # Forward pass
        outputs = model(pixel_values=pixel_values, labels=labels)
        
        # SegFormer calculates cross-entropy loss automatically if labels are provided
        loss = outputs.loss
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        epoch_loss += loss.item()
        progress_bar.set_postfix({"Loss": f"{loss.item():.4f}"})

    # === 1. Run Validation Phase ===
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for batch in val_loader:
            pixel_values = batch["pixel_values"].to(device)
            labels = batch["labels"].to(device)
            
            outputs = model(pixel_values=pixel_values, labels=labels)
            val_loss += outputs.loss.item()
            
    avg_val_loss = val_loss / len(val_loader)
    avg_train_loss = epoch_loss / len(train_loader)
    
    # === 2. Update the progress bar display cleanly ===
    progress_bar.set_postfix({
        "Train Loss": f"{avg_train_loss:.4f}", 
        "Val Loss": f"{avg_val_loss:.4f}"
    })
    
    print(f"\nEpoch {epoch+1} Complete. Train: {avg_train_loss:.4f} | Val: {avg_val_loss:.4f}")
        
    #print(f"Epoch {epoch+1} Complete. Average Train Loss: {epoch_loss / len(train_loader):.4f}")
    
    # Save a checkpoint after each epoch
    torch.save(model.state_dict(), f"segformer_uavid_epoch_{epoch+1}.pt")

print("Training finished successfully!")