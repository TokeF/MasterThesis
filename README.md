# Automatic Detection of Electromagnetic Couplings

This repository contains the codebase for a master's thesis project focused on developing machine learning methods for the automatic detection of electromagnetic couplings in transient electromagnetic (TEM) data. The project explores various machine learning models, including random forests and autoencoders, to improve the efficiency of identifying and culling couplings in TEM data.

## Abstract

The transient electromagnetic method is a well-established geophysical technique for subsurface delineation, commonly used in mineral exploration, groundwater exploration, and environmental mapping. A new instrument for high-resolution mapping of resistivity structures in the shallow subsurface has recently been developed. However, the system is sensitive to electromagnetic coupling with man-made conductors, requiring significant data processing time to identify and remove couplings.

This thesis investigates machine learning-based methods for efficient automatic detection of couplings. Starting with a simple random forest model, limitations and challenges were identified, leading to the development of a more advanced unsupervised one-class autoencoder neural network. The final model achieved approximately 90% hit rate, demonstrating the potential for fully automated coupling detection using synthetic data.

## Project Structure

The repository is organized as follows:

- **`AutoSK/`**: Contains code for AutoSklearn-based classification.
- **`lstm/`**: Includes implementations of LSTM models for time-series data.
- **`MLone/`**: Scripts for training and evaluating machine learning models, including visualization and saving results.
- **`NN/`**: Neural network models, including autoencoders, CNNs, and ResNet implementations.
- **`SVM/`**: SVM and random forest models, along with TPOT-exported pipelines.
- **`utilities/`**: Utility scripts for data preprocessing, visualization, metrics calculation, and more.

## Key Features

- **Random Forest Models**: Initial exploration of machine learning for coupling detection.
- **Autoencoder Neural Networks**: Advanced unsupervised models for one-class classification.
- **Data Preprocessing**: Tools for handling and normalizing TEM data.
- **Visualization**: Scripts for plotting results, including ROC curves and misclassified samples.
