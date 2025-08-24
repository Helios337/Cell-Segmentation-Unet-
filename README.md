🔬 U-Net Cell Segmentation & Counting Tool

An end-to-end deep learning tool to automatically identify and count cells in microscopy images. This project automates the tedious task of manual cell counting, providing fast, accurate, and reproducible results for researchers and enthusiasts.

At its core, it uses a powerful U-Net neural network to find the cells and a clever watershed algorithm to separate any that are clustered together, ensuring a precise count.
✨ Key Features

    🧠 Deep Learning Core: Built with a classic U-Net architecture in TensorFlow/Keras, a proven model for biomedical image segmentation.

    🧩 Smart Counting: Goes beyond just finding cells. It uses a post-processing pipeline with the watershed algorithm to intelligently separate overlapping cells, leading to more accurate counts.

    🧪 Synthetic Data Included: Comes with a data generator to create cell-like images on the fly, so you can run a full training and prediction demo right out of the box.

    📊 Clear Visualizations: Automatically generates a visual report for its predictions, showing the original image, the ground truth, the model's raw output, and the final labeled cells with a count.

    modular, and easy-to-read Python code.

🚀 Getting Started

Ready to see it in action? Follow these steps to get the project running on your local machine.
1. Prerequisites

Make sure you have Python (version 3.9 or newer) and pip installed on your system.
2. Installation

First, clone this repository to your local machine using Git:

git clone https://github.com/Helios337/Cell-Segmentation-Unet-.git
cd Cell-Segmentation-Unet-

Next, it's highly recommended to create and activate a virtual environment. This keeps the project's dependencies neatly isolated.

# Create the virtual environment
python -m venv venv

# Activate it
# On Windows:
.\venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate

Finally, install all the required libraries from the requirements.txt file:

pip install -r requirements.txt

3. Running the Demonstration

With everything installed, you can now run the main script. This will kick off the full demonstration using synthetic data.

python main.py

The script will:

    Build the U-Net model architecture.

    Generate a fresh set of synthetic cell images for training.

    Train the model (you'll see the progress live!).

    Evaluate the trained model on a test set.

    Pick a random test image and display a 4-panel plot showing the segmentation results.

    Show the training history graphs for loss and accuracy.

📂 Project Structure

The codebase is organized into logical modules to make it easy to understand and build upon.

.
├── 📄 main.py             # The main script to run the entire pipeline
├── 🧠 model.py             # Contains the CellSegmentationTool class (the U-Net)
├── 💾 data_handler.py      # Handles data loading and synthetic data generation
├── 🛠️ utils.py              # Extra helper functions
├── 📋 requirements.txt    # A list of all the Python libraries needed
└── 📜 README.md           # This file!

🤝 Contributing

Contributions are welcome! If you have ideas for improvements, feel free to fork the repository, make your changes, and submit a pull request. Here are a few ideas:

    Train on Real Data: The data_handler.py includes a function to download data from the Broad Bioimage Benchmark Collection (BBBC). Try training the model on a real-world dataset like BBBC005!

    Build a User Interface: Wrap the tool in a simple GUI using Streamlit or Gradio to allow users to upload their own images for counting.

    Experiment with Models: Could a different model architecture like Mask R-CNN or a Vision Transformer perform even better?

If you run into any issues or have a question, please open an issue on GitHub.
