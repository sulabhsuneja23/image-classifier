# 🧠 Image Classifier using Keras/TensorFlow

This project is a simple image classifier built using Convolutional Neural Networks (CNNs) in TensorFlow/Keras. It classifies images into 4 categories based on a synthetic dataset.

---

## 🚀 Features

- Trains a Keras CNN to classify the images based on a pre-defined dataset
- Predicts the class of new/unseen images based on the model trained above
- Supports batch prediction on folders for the test data
- Easy to extend to real-world datasets

---

## 📦 Requirements

Make sure Python 3.11 or less is installed (as tensorflow is not available for python 3.12 >= ), then install the dependencies:

```bash
pip install -r requirements.txt
```
## How to Run

To train the model and save the model, run the following command:

```bash
python classifier.py
```

To predict the class for the test images, run following command:

```bash
python predict.py
```

#### Note

`dataset/` includes 100 sample images divided equally into 4 classes `['class1', 'class2', 'class3', 'class4']`
`testdata/` includes 10 images of different classes.
Play with the model and try tweaking the values of `epochs` and `batch_size` in classifier.py and observe the `accuracy` and `loss`.
