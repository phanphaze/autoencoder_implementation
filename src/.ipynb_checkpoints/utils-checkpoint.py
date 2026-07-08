import matplotlib.pyplot as plt
import torch
from src.config import model_save_dir
from src.model import Autoencoder, ConvAutoencoder
from src.dataset import get_train_loader

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

# Determines the sequential filename for the current training session.
def get_next_version_name(architecture="conv"):
    model_type = "conv" if architecture == "conv" else "linear"
    existing_files = list(model_save_dir.glob(f"{model_type}_autoencoder_*.pth"))
    
    if not existing_files:
        return f"{model_type}_autoencoder_1.pth"
    
    versions = [int(f.stem.split('_')[-1]) for f in existing_files]
    return f"{model_type}_autoencoder_{max(versions) + 1}.pth"

# Saves the model state dictionary to disk.
def save_model(model, filename):
    model_save_dir.mkdir(parents=True, exist_ok=True)
    save_path = model_save_dir / filename
    
    torch.save(model.state_dict(), save_path)
    print(f"Model saved to: {save_path}")

# Automatically loads the correct model architecture and weights, then plots the reconstructions.
def evaluate_and_plot(architecture="conv", version=None):
    # 1. Instantiate the correct blueprint
    model = ConvAutoencoder() if architecture == "conv" else Autoencoder()
    model_type = "conv" if architecture == "conv" else "linear"
    
    # 2. Find the file to load
    if version:
        filename = f"{model_type}_autoencoder_{version}.pth"
    else:
        # If no version provided, auto-find the latest one
        files = list(model_save_dir.glob(f"{model_type}_autoencoder_*.pth"))
        if not files:
            print(f"No saved models found for architecture: {architecture}")
            return
        
        # Sort files by their version number and grab the highest
        latest_file = max(files, key=lambda f: int(f.stem.split('_')[-1]))
        filename = latest_file.name

    # 3. Load the weights
    save_path = model_save_dir / filename
    try:
        model.load_state_dict(torch.load(save_path))
        print(f"Successfully loaded {filename}")
    except Exception as e:
        print(f"Failed to load weights. Did the architecture change?\nError: {e}")
        return

    # 4. Plot
    loader = get_train_loader()
    plot_reconstruction(model, loader)

# Takes a list of average loss values and plots them against the epoch number.
def plot_loss_curve(epoch_losses):
    # Create a clean canvas
    plt.figure(figsize=(8, 5))
    
    # Plot the data with a solid line and distinct dots for each epoch
    plt.plot(epoch_losses, marker='o', linestyle='-', color='b', label='Training Loss')
    
    # Label the axes to make the visual clear
    plt.title("Model Training Loss over Epochs")
    plt.xlabel("Epoch")
    plt.ylabel("Average MSE Loss")
    
    # Add a subtle grid to help estimate values visually
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    
    # Render the graph
    plt.tight_layout()
    plt.show()