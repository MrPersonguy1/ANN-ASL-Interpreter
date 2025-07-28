🤟 ANN American Sign Language (ASL) Interpreter
This project uses an Artificial Neural Network (ANN) to recognize hand signs from the American Sign Language (ASL) alphabet using image data. It focuses on classifying letters from images of hands signing them.

📚 What You’ll Learn
How to load image data for machine learning

How to build a basic neural network using PyTorch

How to train a model on ASL sign images

How to test the model and check its accuracy

How to visualize training loss and model predictions

🛠️ Technologies Used
Python 🐍

PyTorch (for building the ANN)

pandas and NumPy (for data handling)

Matplotlib (for plotting)

KaggleHub (to load datasets easily)

🧠 How It Works
Image data is loaded from the ASL MNIST dataset on Kaggle.

Each image is reshaped into a 28x28 grayscale format.

A dictionary is created to match numeric labels to letters (A–Z, skipping J).

An MLP (Multilayer Perceptron) model is built using PyTorch.

The model is trained over 10 epochs using a training loop.

The training loss is plotted to see how the model improves.

The model is tested on unseen data to measure accuracy.

You can select a test image and see what letter the model predicts!

🚀 How to Run
Make sure Python and PyTorch are installed.

Install required libraries:

bash
Copy
Edit
pip install pandas numpy matplotlib torch torchvision kagglehub
Run the script in a Jupyter Notebook or Google Colab (recommended).

The dataset is downloaded automatically from Kaggle using kagglehub.

Run the cells step-by-step to train and test your model.

📁 File Overview
ANN_ASL_Interpreter.py: Full code for loading the data, building the ANN, training the model, testing accuracy, and visualizing results.
