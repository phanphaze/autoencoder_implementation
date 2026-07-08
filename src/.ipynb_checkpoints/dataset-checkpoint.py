import pandas as pd
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from src.config import raw_data_dir, processed_data_path, batch_size

class HandwritingDataset(Dataset):
    def __init__(self, dataframe, transform=None):
        self.df = dataframe
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
         # Get the path from dataframe
        img_path = self.df.iloc[idx]['filepath']
        
        # Load the image (L mode ensures it is grayscale)
        image = Image.open(img_path).convert('L')

        # Apply the transform
        if self.transform:
            image = self.transform(image)
            
        return image, image #the label is the original image in an autoencoder

def get_train_loader():
    df = pd.read_csv(processed_data_path)
    transform = transforms.Compose([
        transforms.Resize((28, 28)),
        transforms.Grayscale(num_output_channels=1),
        transforms.ToTensor()
    ])
    dataset = HandwritingDataset(df, transform=transform)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)

def get_balanced_df():
    # Scans folders and returns a balanced dataframe
    data_list = []
    for folder in raw_data_dir.iterdir():
        if folder.is_dir():
            for file_path in folder.glob("*.jpg"):
                data_list.append({"filepath": str(file_path), "label": int(folder.name)})
    
    df = pd.DataFrame(data_list)
    min_count = df['label'].value_counts().min()
    return df.groupby('label').apply(lambda x: x.sample(n=min_count), include_groups=False).reset_index(drop=True)