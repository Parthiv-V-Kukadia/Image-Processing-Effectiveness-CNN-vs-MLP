#   Image Processing Effectiveness: CNN vs. MLP for Character Recognition

This repository contains the MATLAB code developed for a project comparing the effectiveness of a Convolutional Neural Network (CNN) and a Multi-Layer Perceptron (MLP) for character recognition, following a series of image pre-processing steps. The project aims to classify the characters present in a microchip label image.

##   Project Structure

The repository includes the following MATLAB files:

* **`mlp.m`:** Implements and trains a Multi-Layer Perceptron (MLP) neural network for character classification.
* **`cnn.m`:** Implements and trains a Convolutional Neural Network (CNN) for character classification.
* **`Implementation.m`:** Contains the main script that performs the image pre-processing steps on the microchip label image and then uses both the trained CNN and MLP models to classify the segmented characters.
* **`p_dataset_26.zip`:** (This file is expected to be present in the same directory or accessible path as defined in the scripts) Contains the dataset of character images used for training and testing the CNN and MLP models. The dataset is divided into `train` and `test` subfolders, with character classes as folder names.

##   Project Description

The `Implementation.m` script performs the following sequence of image processing tasks on the `charact2.bmp` image (expected to be in the same directory):

1.  **Contrast Enhancement:** Displays the original image and experiments with various contrast enhancement techniques (imadjust, histeq, adapthisteq).
2.  **Averaging Filter:** Implements and applies a 5x5 averaging filter for image smoothing, also experimenting with different filter sizes.
3.  **High-Pass Filter:** Applies a high-pass filter in the frequency domain to enhance edges.
4.  **Sub-Image Creation:** Extracts a sub-image containing the "HD44780A00" line.
5.  **Binary Conversion:** Converts the sub-image to a binary image.
6.  **Character Outline Detection:** Determines the outlines of the characters in the binary sub-image.
7.  **Character Segmentation:** Segments the image to isolate individual characters.
8.  **CNN and MLP Classification:**
    * Loads pre-trained CNN (`cnn.mat`) and MLP (`mlp.mat`) models (trained using the `p_dataset_26` dataset).
    * Pre-processes the segmented characters to a consistent size (128x128).
    * Uses both models to predict the class of each segmented character.
    * Displays the segmented characters with their predicted labels and confidence scores, indicating whether the prediction was correct based on the expected sequence "HD44780A00".
9.  **Rotation Test:** Rotates one of the segmented characters and tests the classification performance of both the CNN and MLP on the rotated image.

The `mlp.m` and `cnn.m` scripts handle the training of the respective classification models using the provided `p_dataset_26` dataset. They perform image loading, pre-processing (binarization, resizing), define the network architectures, set training parameters, train the models, evaluate their performance on the test set, and save the trained networks.

##   Setup

1.  **Download the Repository:** Clone or download this repository to your local machine.
2.  **Download the Dataset:** Ensure that the `p_dataset_26.zip` file is downloaded and extracted into the same directory as the MATLAB scripts, resulting in a `p_dataset_26` folder with `dataset\train` and `dataset\test` subfolders.
3.  **Place the Image:** Place the `charact2.bmp` image file in the same directory as the MATLAB scripts.
4.  **MATLAB Environment:** Ensure you have MATLAB installed, along with the Image Processing Toolbox and Neural Network Toolbox (Deep Learning Toolbox for CNN). For GPU acceleration of CNN training, a compatible NVIDIA GPU and the Parallel Computing Toolbox are recommended (as indicated in `cnn.m`).

##   Usage

1.  **Run `mlp.m`:** Execute this script in MATLAB to train the MLP model. The trained network will be saved as `mlp.mat`.
2.  **Run `cnn.m`:** Execute this script in MATLAB to train the CNN model. The trained network will be saved as `cnn.mat`. This step might take a significant amount of time depending on your hardware.
3.  **Run `Implementation.m`:** Execute this script in MATLAB to perform the image pre-processing steps on `charact2.bmp`, segment the characters, and classify them using the pre-trained CNN and MLP models. The script will display various stages of image processing and the final classification results for both methods.

##   Results

The `Implementation.m` script will output figures showing:

* Original image and contrast-enhanced versions.
* Image after applying averaging filters of different sizes.
* Image after applying high-pass filters with different cut-off frequencies.
* The segmented sub-image.
* The binary version of the sub-image.
* The outlined characters.
* The segmented individual characters.
* The classification results for each segmented character using both the MLP and CNN models, along with confidence scores and correctness indicators.
* The classification results on a rotated character for both models.

##   Discussion

A report provided contains a detailed discussion of:

* An introduction to the character recognition problem and the chosen methods.
* A description of the algorithms used for image pre-processing, CNN, and MLP.
* Screen captures of each stage of the image processing pipeline.
* A comparison of the effectiveness and efficiency of the CNN and MLP approaches for this task, including an analysis of any differences in their performance.
* An explanation of the rationale behind the chosen methods and any investigations or hyperparameter tuning performed.
* Lessons learned during the project.

##   Note

* This repository contains the code implementation as requested by the assignment. The accompanying report, as mentioned in the assignment, will provide the detailed analysis, explanations, and screen captures.
* The dataset `p_dataset_26.zip` is available under the code section.
* The pre-trained models (`mlp.mat` and `cnn.mat`) will be saved after running `mlp.m` and `cnn.m` respectively. If these files are not present, `Implementation.m` will attempt to load them, potentially leading to errors. **Ensure you run the training scripts (`mlp.m` and `cnn.m`) before running `Implementation.m`.**
* The performance of the CNN might be better if trained for more epochs or with a larger and more diverse dataset, and with further hyperparameter tuning.
