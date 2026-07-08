import torch
import torch.nn as nn
from src.dataset import get_train_loader
from src.model import Autoencoder
from src.utils import save_model
import src.config as config

def train_model():
    # grab the processed data
    train_loader = get_train_loader()
    
    # Setup Training
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") #run on gpu if possible
    print(f"Training on device: {device}")
    
    model = Autoencoder().to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config.Learning_rate)

    # 4. Training Loop
    for epoch in range(config.num_epochs):
        model.train() # Set to training mode
        for images, _ in train_loader:
            images = images.to(device) 
            
            reconstructed = model(images)
            loss = criterion(reconstructed, images)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
        print(f"Epoch {epoch}, Loss: {loss.item():.4f}")
        
    save_model(model, "handwriting_autoencoder.pth")

if __name__ == "__main__":
    train_model()