import pandas as pd
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
from PIL import Image
import src.config as config

from src.config import processed_data_path, batch_size

class HandwritingDataset(Dataset):
    def __init__(self, dataframe, transform=None):
        self.df = dataframe
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        img_path = self.df.iloc[idx]['filepath']
        image = Image.open(img_path).convert('L')

        if self.transform:
            image = self.transform(image)
            
        return image, image

# Reads the CSV, splits the data, and returns train and validation loaders.
def get_dataloaders():

    df = pd.read_csv(processed_data_path)
    transform = transforms.Compose([
        transforms.Resize((28, 28)),
        transforms.Grayscale(num_output_channels=1),
        transforms.ToTensor()
    ])
    
    full_dataset = HandwritingDataset(df, transform=transform)
    
    # Calculate split sizes
    train_size = int(config.train_test_split * len(full_dataset))
    val_size = len(full_dataset) - train_size
    
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False)
    
    return train_loader, val_loader