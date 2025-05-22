# Human Face Generation
This project implements a **Generative Adversarial Network (GAN)** that learns to synthesize realistic 64×64 RGB images of human faces. The model is trained on a face dataset and progressively learns to generate believable, high-quality outputs using adversarial training.

**Note: This project is still in development. The results in this README are not final.**

## Dataset
The dataset used for this project can be downloaded from [Kaggle](https://www.kaggle.com/datasets/greatgamedota/ffhq-face-data-set).

## Project Structure
```
.
├── dataset/
│   └── images128x128/        # Source image dataset (cropped face images)
├── weights/                  # Checkpoints and generated samples (e.g., epoch_50.png)
├── src/
│   ├── Dataloader.py         # Custom dataset for loading and resizing face images
│   ├── GAN.py                # Generator and Discriminator model definitions
│   └── Train.py              # Training loop with image saving and checkpointing
├── Run.py                    # Entry point for training
├── requirements.txt
└── README.md
```

## Installation
To run this project, clone this repository and install the necessary dependencies. You can use the following command to install the required libraries:
```
pip install -r requirements.txt
```

### Requirements
- torch
- torchvision

### Prepare Dataset
Download and extract the dataset into:
```
dataset/images128x128/
```

## How to Run
1. Clone this repository or download the necessary files.
2. Run the Python script `Run.py` to train and evaluate the models.
```
python Run.py
```
The script will train the GAN for the specified number of epochs. A checkpoint as well as 16 randomly generated faces from the model will be saved every 10 epochs.

## Results
The script will output the losses on each epoch of the disciminator and the generator. As of now, the network has only been trained up to epoch 50.

Image Quality at Epoch 50:
- Faces have distinct identities
- Lighting and contrast are better balanced
- Some faces are surprisingly well-structured: clear eyes, smiles, facial symmetry
- Minor flaws:
  - Some samples show warping, especially around glasses or edges
  - A few blended features (e.g., merged eyes or hairlines), typical for 64×64 resolution
 
<img src="https://github.com/user-attachments/assets/b6370149-d2d8-47f6-b2ba-5af782f31438" alt="50 Epochs" width="500" height="500">
