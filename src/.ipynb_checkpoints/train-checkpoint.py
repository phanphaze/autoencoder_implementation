import torch
import torch.nn as nn

from src.dataset import get_dataloaders
from src.model import Autoencoder, ConvAutoencoder
from src.utils import save_model, get_next_version_name
import src.config as config

def train_model(architecture="conv", patience=config.patience, min_delta=config.min_delta):
    # Model selection
    if architecture == "conv":
        model = ConvAutoencoder()
        print("Model: Autoencoder with CNN")
    else:
        model = Autoencoder()
        print("Model: Linear Autoencoder")
        
    train_loader, val_loader = get_dataloaders()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on device: {device}")
    
    model = model.to(device)
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config.Learning_rate)

    # Determine the sequential filename for this training run
    save_filename = get_next_version_name(architecture)
    
    epoch_train_losses = []
    epoch_val_losses = []
    epochs_no_improve = 0

    for epoch in range(config.num_epochs):
        model.train() 
        running_train_loss = 0.0
        
        for images, _ in train_loader:
            images = images.to(device) 
            
            reconstructed = model(images)
            loss = criterion(reconstructed, images)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            running_train_loss += loss.item()
            
        avg_train_loss = running_train_loss / len(train_loader)
        epoch_train_losses.append(avg_train_loss)

        
        model.eval() # Freeze dropout/batchnorm layers
        running_val_loss = 0.0
        
        with torch.no_grad(): # Don't build the gradient graph
            for images, _ in val_loader:
                images = images.to(device)
                reconstructed = model(images)
                loss = criterion(reconstructed, images)
                running_val_loss += loss.item()
                
        avg_val_loss = running_val_loss / len(val_loader)
        epoch_val_losses.append(avg_val_loss)
        
        print(f"Epoch {epoch} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")
        
        # --- 3. EARLY STOPPING (Now tracking Validation Loss) ---
        if epoch > 0:
            # We care if the model stops generalizing, so we check val_loss
            loss_change = abs(epoch_val_losses[-2] - avg_val_loss)
            
            if loss_change < min_delta:
                epochs_no_improve += 1
            else:
                epochs_no_improve = 0
                
            if epochs_no_improve >= patience:
                print(f"Early stopping triggered at epoch {epoch} due to validation stagnation.")
                break
    
    save_model(model, save_filename)
    
    # Return both lists so we can plot them
    return epoch_train_losses, epoch_val_losses

if __name__ == "__main__":
    train_model()