# Image Classification Using Deep Learning (CNN) + Introduction to JAX

## Objective
Training a convolutional neural network (**CNN**) to classify ship images into 5 categories, along with a practical introduction to the **JAX** library for computationally accelerated mathematics.

## Content

### 1. Ship Classification 🗲 CNN
- Dataset: [Game of Deep Learning: Ship Datasets](https://www.kaggle.com/datasets/arpitjain007/game-of-deep-learning-ship-datasets)
- Automatic organization of images into directories by class (`Cargo`, `Military`, `Carrier`, `Cruise`, `Tankers`)
- Building an image data pipeline using `keras.utils.image_dataset_from_directory`
- Training a convolutional neural network for multi-class classification
- Testing the model on new images along with visualization of prediction results

### 2. Introduction to JAX 🗲 Numerical Computing
- Vector operations using `jax.numpy` — comparing syntax and performance with the NumPy library
- Practical use of GPU hardware acceleration via JAX 🗲 a key concept in modern machine learning

## Technologies
`Python` `TensorFlow/Keras` `JAX` `NumPy` `matplotlib`

## How to Run
```bash
pip install tensorflow jax matplotlib kaggle
jupyter notebook "a_rozpoznawanie statkow.ipynb"
jupyter notebook a_jax.ipynb
