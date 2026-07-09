import pandas as pd
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image

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
            
        return image

# Reads the pre-processed CSV and returns a PyTorch DataLoader.
def get_train_loader():
    df = pd.read_csv(processed_data_path)
    transform = transforms.Compose([
        transforms.Resize((28, 28)),
        transforms.Grayscale(num_output_channels=1),
        transforms.ToTensor()
    ])
    dataset = HandwritingDataset(df, transform=transform)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)