# Handwriting Autoencoder Implementation

A modular, research-oriented PyTorch pipeline designed to train and evaluate autoencoders for handwriting reconstruction. This project isolates training, data processing, and architecture definitions for improved reproducibility and experiment tracking.

## Project Structure

* **`data/`**: Raw image datasets and processed CSV manifests.
* **`models/`**: Saved model checkpoints (`.pth`). *[Ignored by Git]*
* **`notebooks/`**: Prototyping and visualization tools. *[Ignored by Git]*
* **`src/`**: Core source code.
    * `config.py`: Hyperparameters and absolute pathing.
    * `dataset.py`: Data loading and transformation logic.
    * `model.py`: Autoencoder architecture definitions (`Autoencoder` and `ConvAutoencoder`).
    * `train.py`: Training loop with stagnation detection and early stopping.
    * `utils.py`: Visualization and evaluation tools.
    * `preprocessing.py`: Installs dataset from Kragglehub, moves it into the proper directory and converts it into an interpreatble format.
* **`.gitignore`**: Git exclusion rules to prevent storage bloat.
* **`environment.yml`**: Conda environment blueprint.
* **`README.md`**: Project documentation.

## Getting Started

### 1. Environment Setup
Recreate the exact Conda environment used for this project to ensure dependency stability:
```bash
conda env create -f environment.yml
conda activate autoencoder_intro
```

### 2. Data Preparation
This project uses the Kagglehub API to download the raw images. Run the preprocessing script to automatically download the dataset, organize the folders, and generate the training manifest:

```bash
python -m src.preprocess
```

### 3. Training
Execute the training loop from the root directory. The script automatically tracks the validation loss and will actively overwrite a `best.pth` file during training, preventing storage bloat while guaranteeing you retain the optimal weights.

```bash
# Train the Convolutional Autoencoder (Default)
python -m src.train --architecture conv

# Train the Linear Autoencoder
python -m src.train --architecture linear
```

### 4. Evaluation
Use the `evaluate_and_plot()` utility within your Jupyter Notebook to instantly load the correct architecture and visualize the reconstructed handwritten digits against the original data. 
```python
from src.utils import evaluate_and_plot

# Automatically evaluates the best Convolutional model
evaluate_and_plot(architecture="conv")
```

## Key Features
* **Modular Architecture:** Easily swap between `Autoencoder` and `ConvAutoencoder` classes via command line or function arguments.
* **Smart Saving:** Overwrites a single `best.pth` file during gradient descent, ensuring the final saved model reflects the lowest loss prior to any stagnation.
* **Early Stopping:** Halts gradient descent upon flatlining to optimize compute resources.

---
**Author:** Cadence Centers
*Developed for the SkAI Institute SEED Program | Illinois Institute of Technology*