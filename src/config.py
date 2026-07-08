from pathlib import Path

# Hyperparameters
Learning_rate = 1e-4
batch_size = 32
num_epochs = 25
latent_dimension_size = 32
train_test_split = 0.7
min_delta = 0.0001
project_root = Path(__file__).resolve().parent.parent

# Data paths
raw_data_dir = project_root / "data" / "raw"
processed_data_path = project_root / "data" / "processed" / "processed.csv"
model_save_dir = project_root / "models"