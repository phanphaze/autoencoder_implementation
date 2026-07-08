import matplotlib.pyplot as plt
import torch
from src.config import model_save_dir

# Plots the original images side-by-side with their reconstructions.
def plot_reconstruction(model, loader, device="cpu"):
    model.eval()
    images, _ = next(iter(loader)) # Get a batch
    images = images.to(device)
    
    with torch.no_grad():
        reconstructed = model(images)
    
    # move to CPU for plotting
    images = images.cpu()
    reconstructed = reconstructed.cpu()
    
    # Plotting
    fig, axes = plt.subplots(2, 5, figsize=(10, 4))
    for i in range(5):
        axes[0, i].imshow(images[i].squeeze(), cmap='gray')
        axes[0, i].axis('off')
        axes[0, i].set_title("Original")
        
        axes[1, i].imshow(reconstructed[i].squeeze(), cmap='gray')
        axes[1, i].axis('off')
        axes[1, i].set_title("Recon")
    
    plt.tight_layout()
    plt.show()

def save_model(model, filename="autoencoder_best.pth"):
    # Ensure the directory exists before saving
    model_save_dir.mkdir(parents=True, exist_ok=True)
    save_path = model_save_dir / filename
    torch.save(model.state_dict(), save_path)
    print(f"Model saved to: {save_path}")