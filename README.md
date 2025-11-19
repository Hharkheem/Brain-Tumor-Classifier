# Brain Tumor Classification

This repository contains a Python application for classifying brain tumor images using a Convolutional Neural Network (CNN) built with PyTorch and a user interface created with Gradio.

## Project Overview

The project implements a CNN model to classify brain tumor images into four categories: **glioma**, **meningioma**, **notumor**, and **pituitary**. The model is trained on a dataset of brain MRI images and provides a web-based interface for users to upload images and receive classification predictions.

## Repository Structure

- **app.py**: The main application script that loads the trained model and launches a Gradio interface for image classification.
- **main.ipynb**: A Jupyter Notebook containing the code for data preprocessing, model training, evaluation, and visualization of results.
- **model1.pt**: The trained model weights saved after training.
- **data/**: Directory containing the training and testing datasets (https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset?resource=download).

## Requirements

To run this project, you need the following dependencies:

- Python 3.8+
- PyTorch
- torchvision
- Gradio
- Pillow
- Matplotlib

Install the required packages using:

```bash
pip install torch torchvision gradio pillow matplotlib
```

## Dataset

The dataset used for training and testing is expected to be organized in the following structure:

```
data/
├── Training/
│   ├── glioma/
│   ├── meningioma/
│   ├── notumor/
│   └── pituitary/
└── Testing/
    ├── glioma/
    ├── meningioma/
    ├── notumor/
    └── pituitary/
```

Each subdirectory should contain MRI images corresponding to the respective class. The dataset is not included in this repository due to its size. You can obtain a suitable dataset (e.g., Brain Tumor MRI Dataset) from sources like Kaggle or other open data repositories.

## Usage

### Training the Model

1. Ensure the dataset is placed in the `data/` directory as described above.
2. Open `main.ipynb` in a Jupyter Notebook environment.
3. Run the cells sequentially to:
   - Load and preprocess the dataset.
   - Define and train the CNN model.
   - Evaluate the model on the test set.
   - Save the trained model weights as `model1.pt`.

The model achieves approximately **97.33% accuracy** on the test set after 30 epochs, as shown in the notebook.

### Running the Application

1. Ensure the trained model file (`model1.pt`) is in the same directory as `app.py`.
2. Run the Gradio interface:

```bash
python app.py
```

3. Open the provided URL in your browser to access the Gradio interface.
4. Upload an MRI image to classify it into one of the four classes.

## Model Architecture

The CNN model consists of the following layers:

- 3 Convolutional layers with ReLU activation and MaxPooling.
- A Flatten layer to transition to fully connected layers.
- A fully connected layer with 256 units and ReLU activation.
- A Dropout layer (0.5 probability) for regularization.
- A final fully connected layer outputting probabilities for 4 classes.

The model is trained using the Adam optimizer with a learning rate of 0.0001 and Cross-Entropy Loss.

## Results

The training process is detailed in `main.ipynb`, with the loss decreasing over 30 epochs and a final test accuracy of **97.33%**. The notebook also includes a visualization of a random test image with its predicted and actual class labels, along with the confidence score.
<img width="1354" height="684" alt="image" src="https://github.com/user-attachments/assets/79be7021-b384-4ea9-bd7d-37fe87b947f8" />

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
