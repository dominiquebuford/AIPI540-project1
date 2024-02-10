import pandas as pd
import numpy as np
import os
import sys
#import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report
from skimage.io import imread
from skimage.io import ImageCollection
from skimage.transform import resize

# Function to load and preprocess images from a folder
def load_images_from_folder(folder):
    images = []
    labels = []
    for filename in os.listdir(folder):
        path_to_imgs = os.path.join(folder, filename)
        for image in os.listdir(path_to_imgs):
          single_img_path = os.path.join(path_to_imgs, image)
          img = cv2.imread(single_img_path)
          if img is not None:
              # Preprocess the image (e.g., resize, normalize, convert to grayscale)
              img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
              img = cv2.resize(img, (64, 64))  # Resize to a fixed size
              images.append(img)
              labels.append(single_img_path.split('/')[-2])  # Assuming folder name is the class label
    return images, labels

# Load images and labels from train, test, and val folders
train_images, y_train = load_images_from_folder('/content/Vegetable Images/train')
test_images, y_test = load_images_from_folder('/content/Vegetable Images/test')
val_images, y_val = load_images_from_folder('/content/Vegetable Images/validation')

# Extract features from images (e.g., flatten the image)
X_train = np.array([img.flatten() for img in train_images])
X_test = np.array([img.flatten() for img in test_images])
X_val = np.array([img.flatten() for img in val_images])

# Define SVC
model = SVC(kernel='rbf', C=10)
model.fit(X_train, y_train)

# Predict on validation set
preds_val = model.predict(X_val)
val_acc = accuracy_score(y_val, preds_val)

# Predict on test set
test_preds = model.predict(X_test)
acc_test = accuracy_score(y_test, test_preds)

# Print the accuracy
print(f'Validation accuracy: {val_acc}')
print(f'Test accuracy: {acc_test}')