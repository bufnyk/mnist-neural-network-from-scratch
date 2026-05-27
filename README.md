# NumPy Neural Network from Scratch (MNIST)

A custom, two-layer neural network built entirely from scratch using only NumPy and basic Python libraries. This project demonstrates the core mathematical concepts behind deep learning (forward propagation, backpropagation, activation functions) without relying on high-level frameworks like TensorFlow or PyTorch.

The model is trained on the classic MNIST dataset to recognize handwritten digits (0-9) and includes an interactive visualizer to test the model's predictions.

---

## 🚀 Features

* **Zero Frameworks:** The entire neural network architecture, including gradient descent and backpropagation math, is written from scratch.
* **Custom Architecture:** Features an input layer (784 nodes), a hidden layer (128 nodes) using the ReLU activation function, and an output layer (10 nodes) using Softmax.
* **Training Pipeline:** Automatically calculates loss, updates weights/biases, and prints accuracy metrics every 10 epochs.
* **Weights Saving:** Successfully trained model weights and biases are exported to a `.npz` file for later use.
* **Interactive Visualizer:** A CLI tool that pulls random digits from the test dataset, runs them through the trained model, and displays the image alongside the AI's prediction using Matplotlib.

---

## 🛠️ Tech Stack

* **Language:** Python
* **Linear Algebra & Math:** NumPy
* **Data Handling:** Pandas
* **Data Visualization:** Matplotlib

---

## 📦 Setup & Installation

To run this project locally, clone the repository and navigate into the directory:

git clone https://github.com/bufnyk/mnist-neural-network-from-scratch.git
cd mnist-neural-network-from-scratch

Install the required dependencies:

pip install numpy pandas matplotlib

Data Preparation:
This project requires the MNIST dataset in CSV format. 
1. Create a folder named data in the root directory.
2. Download mnist_train.csv and mnist_test.csv and place them inside the data folder.

---

## 📂 Project Structure

* train.py: The main script that builds the network architecture, trains the model on the training dataset, and saves the final weights.
* visualize.py: The interactive testing script that loads the saved weights and visualizes predictions on the test dataset.
* data/: Directory containing the MNIST CSV files and the saved model_weights.npz.

---

## 💻 Usage

### 1. Training the Model
To start the training process, run the training script. This will initialize random weights, perform gradient descent for 500 iterations (epochs), and output the accuracy to the console.

python train.py

Once finished, it will automatically save the trained parameters into data/model_weights.npz.

### 2. Testing & Visualization
To test how well the model learned, run the visualizer script:

python visualize.py

The terminal will prompt you to enter an index number. It will then fetch that specific image from the test dataset, run a forward propagation pass using your trained weights, and pop up a Matplotlib window showing the handwritten digit, the expected result, and your model's prediction. Type "stop" to exit the loop.
