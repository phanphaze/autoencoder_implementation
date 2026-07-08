import torch
import torch.nn as nn
from src.dataset import get_train_loader
from src.model import Autoencoder, ConvAutoencoder
from src.utils import save_model, get_next_version_name
import src.config as config

def train_model(architecture="conv", patience=5, min_delta=config.min_delta):
    # Model selection
    if architecture == "conv":
        model = ConvAutoencoder()
        print("Model: Autoencoder with CNN")
    else:
        model = Autoencoder()
        print("Model: Linear Autoencoder")
        
    train_loader = get_train_loader()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on device: {device}")
    
    model = model.to(device)
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config.Learning_rate)

    # Determine the sequential filename for this training run
    save_filename = get_next_version_name(architecture)
    
    epoch_losses = []
    epochs_no_improve = 0

    for epoch in range(config.num_epochs):
        model.train() 
        running_loss = 0.0 
        
        for images, _ in train_loader:
            images = images.to(device) 
            
            reconstructed = model(images)
            loss = criterion(reconstructed, images)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            
        avg_loss = running_loss / len(train_loader)
        epoch_losses.append(avg_loss)
        print(f"Epoch {epoch}, Average Loss: {avg_loss:.4f}")
        
        # Early stopping logic: Assess absolute loss change from the prior epoch
        if epoch > 0:
            loss_change = abs(epoch_losses[-2] - avg_loss)
            
            if loss_change < min_delta:
                epochs_no_improve += 1
            else:
                epochs_no_improve = 0
                
            if epochs_no_improve >= patience:
                print(f"Early stopping triggered at epoch {epoch} due to stagnation.")
                break
    
    # Save the model exactly once at the conclusion of training
    save_model(model, save_filename)
    
    return epoch_losses

if __name__ == "__main__":
    train_model()