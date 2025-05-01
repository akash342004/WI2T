#!/usr/bin/env python
# coding: utf-8

# Importing Libraries
import numpy as np
import tensorflow as tf
from tensorflow import expand_dims
import os
import matplotlib.pyplot as plt
import cv2
import imghdr
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
import warnings
warnings.filterwarnings("ignore")

# PATH ALLOCATION
data_dir = os.path.join(r'Data_language/data')
classes = os.listdir(data_dir)
print("Classes:", classes)

# REMOVING UNWANTED EXTENSIONS
img_ext = ['jpeg', 'jpg', 'png']
for img_c in classes:
    for img_v in os.listdir(os.path.join(data_dir, img_c)):
        img_p = os.path.join(data_dir, img_c, img_v)
        try:
            img_mat = plt.imread(img_p)
            img_tip = imghdr.what(img_p)
            if img_tip not in img_ext:
                print('Image not in ext list {}'.format(img_p))
                os.remove(img_p)
        except Exception as e:
            print(e)

# Load and preprocess data from subfolders
def load_data(folder):
    images = []
    labels = []
    class_names = sorted(os.listdir(folder))
    for class_index, class_name in enumerate(class_names):
        class_path = os.path.join(folder, class_name)
        for image_name in os.listdir(class_path):
            image_path = os.path.join(class_path, image_name)
            image = cv2.imread(image_path)
            if image is None:
                print(f"Skipping corrupt/unreadable image: {image_path}")
                continue
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = cv2.resize(image, (256, 256))
            images.append(image)
            labels.append(class_index)
    return images, labels

# Load and preprocess image data
images, labels = load_data(r'Data_language\data')

# Split data into train and test sets
train_images, test_images, train_labels, test_labels = train_test_split(
    images, labels, test_size=0.2, random_state=42, stratify=labels)

# Convert lists to numpy arrays
train_images = np.array(train_images)
train_labels = np.array(train_labels)
test_images = np.array(test_images)
test_labels = np.array(test_labels)

# Normalize pixel values
train_images = train_images.astype('float32') / 255.0
test_images = test_images.astype('float32') / 255.0

# Print shapes
print("\nData Shapes:")
print("Train images shape:", train_images.shape)
print("Train labels shape:", train_labels.shape)
print("Test images shape:", test_images.shape)
print("Test labels shape:", test_labels.shape)

# Visualizing samples
fig, ax = plt.subplots(ncols=4, figsize=(20, 20))
for i, j in enumerate(train_images[0:4]):
    ax[i].imshow(j)
    ax[i].title.set_text(train_labels[i])
plt.show()

# MODEL BUILDING (EXACTLY AS IN YOUR ORIGINAL CODE)
model = tf.keras.Sequential()
model.add(tf.keras.layers.Conv2D(16, (3,3), 1, activation='relu', input_shape=(256,256,3)))
model.add(tf.keras.layers.MaxPooling2D())
model.add(tf.keras.layers.Conv2D(32, (3,3), 1, activation='relu'))
model.add(tf.keras.layers.MaxPooling2D())
model.add(tf.keras.layers.Conv2D(16, (3,3), 1, activation='relu'))
model.add(tf.keras.layers.MaxPooling2D())
model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(256, activation='relu'))
model.add(tf.keras.layers.Dense(3, activation='sigmoid'))

model.compile('adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Training
history = model.fit(train_images, train_labels, epochs=50, 
                   callbacks=[tf.keras.callbacks.TensorBoard(log_dir='logs')])

# EVALUATION METRICS ADDITION
# [Previous imports and code remain the same until the evaluation metrics section]

# EVALUATION METRICS ADDITION
test_pred = model.predict(test_images)
test_pred_classes = np.argmax(test_pred, axis=1)

# Get the actual number of unique classes in predictions
unique_classes = np.unique(test_pred_classes)
num_classes = len(unique_classes)
print(f"\nDetected {num_classes} classes in predictions: {unique_classes}")

print("\n" + "="*50)
print("MODEL EVALUATION METRICS")
print("="*50)

# Calculate metrics with proper class handling
target_names = ['English', 'Hindi', 'Telugu']
labels_present = [i for i in range(3) if i in unique_classes]  # Only include classes present in predictions
filtered_target_names = [target_names[i] for i in labels_present]

# Calculate metrics
accuracy = accuracy_score(test_labels, test_pred_classes)
precision = precision_score(test_labels, test_pred_classes, average='weighted', labels=labels_present)
recall = recall_score(test_labels, test_pred_classes, average='weighted', labels=labels_present)
f1 = f1_score(test_labels, test_pred_classes, average='weighted', labels=labels_present)

print(f"\nAccuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1-Score: {f1:.4f}")

print("\nClassification Report:")
print(classification_report(test_labels, test_pred_classes, 
                          labels=labels_present,
                          target_names=filtered_target_names))


# Visualization
plt.figure(figsize=(5,5))
plt.plot(history.history['loss'], color="red", label="loss")
plt.title("Training Loss")
plt.show()

plt.figure(figsize=(5,5))
plt.plot(history.history['accuracy'], color="green", label="accuracy")
plt.title("Training Accuracy")
plt.show()

# Sample prediction
ref = {0:"English", 1:"Hindi", 2:"Telugu"}
img = plt.imread(r'D:\AKASH\FINAL YEAR PROJECT WEB BASED IMAGE TO TEXT CONVERSION USING ADVANCED DEEP LEARNING\hindi1.jpg')
plt.imshow(img)
plt.title("Sample Test Image")
plt.show()

resize = tf.image.resize(img, (256,256))
resize = resize/255
img_input = expand_dims(resize, 0)

predictions = model.predict(img_input)
predicted_labels = np.argmax(predictions, axis=1)
print("\nSample Prediction:")
print("The language detected is", ref[predicted_labels[0]])

# Save model
model.save('detector2.keras')
print("\nModel saved as 'detector.keras'")
