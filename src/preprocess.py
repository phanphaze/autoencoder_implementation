import kagglehub
import shutil
import pandas as pd

from src.config import raw_data_dir, processed_data_path

def get_balanced_df():
    """Scans raw folders and returns a balanced dataframe."""
    data_list = []
    for folder in raw_data_dir.iterdir():
        if folder.is_dir():
            for file_path in folder.glob("*.jpg"):
                data_list.append({"filepath": str(file_path), "label": int(folder.name)})
    
    df = pd.DataFrame(data_list)
    min_count = df['label'].value_counts().min()
    return df.groupby('label').apply(lambda x: x.sample(n=min_count), include_groups=False).reset_index(drop=True)

def setup_data():
    print("Downloading dataset from Kaggle...")
    dataset_path = kagglehub.dataset_download("olafkrastoveski/handwrittn-digits-0-9")
    
    print(f"Transferring data to {raw_data_dir}...")
    shutil.copytree(dataset_path, raw_data_dir, dirs_exist_ok=True)

    print("Crawling folders and generating CSV manifest...")
    processed_data_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Generate the map using the function now native to this file
    df = get_balanced_df()
    df.to_csv(processed_data_path, index=False)
    
    print(f"Success! Processed {len(df)} balanced images. Ready for training.")

if __name__ == "__main__":
    setup_data()