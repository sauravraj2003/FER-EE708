import os
import matplotlib.pyplot as plt
import random
import cv2
import shutil
import pandas as pd
from PIL import Image
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import ReduceLROnPlateau
import Augmentor as ag

dataset_path = "/kaggle/input/facial-expression-dataset/train/train"
print(os.listdir(dataset_path))

emotions = os.listdir(dataset_path)
emotion_img_count = {}
for emotion in emotions:
    emotion_path = os.path.join(dataset_path, emotion)
    if os.path.isdir(emotion_path):
        num_imgs = len(os.listdir(emotion_path))
        emotion_img_count[emotion] = num_imgs

for emotion, count in emotion_img_count.items():
    print(f"Emotion: {emotion}, Count: {count}")

x = list(emotion_img_count.keys())
y = list(emotion_img_count.values())
plt.figure(figsize=(10, 5))
plt.bar(x, y, color='green')
plt.xlabel("Emotions")
plt.ylabel("Number of images")
plt.title("Bar diagram to visualize the data distribution")
plt.show()

random_emotion = random.choice(emotions)
random_emotion_path = os.path.join(dataset_path, random_emotion)
img_files = os.listdir(random_emotion_path)
random_img = random.sample(img_files, 5)

fig, axes = plt.subplots(1, 5, figsize=(15, 5))

for i in range(len(random_img)):
    ax = axes[i]
    img_name = random_img[i]
    img_path = os.path.join(random_emotion_path, img_name)
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    ax.imshow(img)
    ax.axis('off')
    ax.set_title(random_emotion)

plt.show()

disgust_path = "/kaggle/input/facial-expression-dataset/train/train/disgust"
disgust_aug_path = "/kaggle/working/disgust_augmented"

if os.path.exists(disgust_aug_path):
    shutil.rmtree(disgust_aug_path)

p = ag.Pipeline(source_directory=disgust_path, output_directory=disgust_aug_path)
p.rotate(probability=0.3, max_left_rotation=10, max_right_rotation=10)
p.crop_random(probability=.1, percentage_area=0.5)
p.flip_top_bottom(probability=0.3)
p.random_brightness(probability=0.5, min_factor=0.4, max_factor=0.9)
p.random_color(probability=0.5, min_factor=0.4, max_factor=0.9)
p.random_contrast(probability=0.5, min_factor=0.9, max_factor=1.4)
p.rotate270(probability=0.3)
p.sample(4000)

surprise_path = "/kaggle/input/facial-expression-dataset/train/train/surprise"
surprise_aug_path = "/kaggle/working/surprise_augmented"

if os.path.exists(surprise_aug_path):
    shutil.rmtree(surprise_aug_path)

p = ag.Pipeline(source_directory=surprise_path, output_directory=surprise_aug_path)
p.rotate(probability=0.3, max_left_rotation=10, max_right_rotation=10)
p.crop_random(probability=.1, percentage_area=0.5)
p.flip_top_bottom(probability=0.3)
p.random_brightness(probability=0.5, min_factor=0.4, max_factor=0.9)
p.random_color(probability=0.5, min_factor=0.4, max_factor=0.9)
p.random_contrast(probability=0.5, min_factor=0.9, max_factor=1.4)
p.rotate270(probability=0.3)
p.sample(1200)

train_dir = "/kaggle/working/train"
test_dir = "/kaggle/working/test"

for directory in [train_dir, test_dir]:
    os.makedirs(directory, exist_ok=True)

for emotion in emotions:
    emotion_path = os.path.join(dataset_path, emotion)
    img_files = os.listdir(emotion_path)
    img_paths = [os.path.join(emotion_path, img) for img in img_files]
    
    train_img, test_img = train_test_split(img_paths, test_size=0.2, random_state=0)
    train_emotion_dir = os.path.join(train_dir, emotion)
    test_emotion_dir = os.path.join(test_dir, emotion)
    os.makedirs(train_emotion_dir, exist_ok=True)
    os.makedirs(test_emotion_dir, exist_ok=True)
    
    for img in train_img:
        shutil.copy(img, train_emotion_dir)
    for img in test_img:
        shutil.copy(img, test_emotion_dir)

disgust_train_dir = os.path.join(train_dir, "disgust")
surprise_train_dir = os.path.join(train_dir, "surprise")

os.makedirs(disgust_train_dir, exist_ok=True)
os.makedirs(surprise_train_dir, exist_ok=True)

def merge_augmented_data(augmented_path, target_path):
    if not os.path.exists(augmented_path):
        print(f"Augmented path {augmented_path} does not exist!")
        return
    
    for file_name in os.listdir(augmented_path):
        src = os.path.join(augmented_path, file_name)
        dst = os.path.join(target_path, file_name)
        shutil.move(src, dst)

merge_augmented_data(disgust_aug_path, disgust_train_dir)
merge_augmented_data(surprise_aug_path, surprise_train_dir)

if os.path.exists(surprise_aug_path):
    shutil.rmtree(surprise_aug_path)

if os.path.exists(disgust_aug_path):
    shutil.rmtree(disgust_aug_path)

def load_dataset(dataset_path):
    images = []
    labels = []
    
    for emotion in os.listdir(dataset_path):
        emotion_path = os.path.join(dataset_path, emotion)
        for image in os.listdir(emotion_path):
            img_path = os.path.join(emotion_path, image)
            images.append(img_path)
            labels.append(emotion)
    
    return images, labels

train_path = "/kaggle/working/train"
test_path = "/kaggle/working/test"
train_df = pd.DataFrame()
train_df['image'], train_df['label'] = load_dataset(train_path)
train_df.head()

img = Image.open(train_df['image'][0])
plt.imshow(img, cmap='gray')
plt.show()

IMG_SIZE = (48, 48)
BATCH_SIZE = 64

train_datagen = ImageDataGenerator(rescale=1./255)
test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    train_path,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    color_mode="grayscale",
    class_mode='categorical',
    shuffle=True
)

test_generator = test_datagen.flow_from_directory(
    test_path,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    color_mode="grayscale",
    class_mode='categorical',
    shuffle=False
)

def load_data(generator):
    X, y = [], []
    for img_batch, label_batch in generator:
        X.append(img_batch)
        y.append(label_batch)
        if len(X) * BATCH_SIZE >= generator.samples:
            break
    return np.concatenate(X), np.concatenate(y)

X_train, y_train = load_data(train_generator)
X_test, y_test = load_data(test_generator)

print(f"X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")
print(f"X_test shape: {X_test.shape}, y_test shape: {y_test.shape}")

train_df = train_df.sample(frac=1, random_state=0)
label_column = train_df[['label']]
encoder = OneHotEncoder(sparse_output=False)
one_hot_encoded = encoder.fit_transform(label_column)
column_names = encoder.get_feature_names_out(['label'])
one_hot_df = pd.DataFrame(one_hot_encoded, columns=column_names)
train_encoded = pd.concat([train_df[['image']], one_hot_df], axis=1)
print(train_encoded.iloc[0])

model = models.Sequential([
    layers.Conv2D(64, (3, 3), activation='relu', padding='same', input_shape=(48, 48, 1)),
    layers.BatchNormalization(),
    layers.MaxPooling2D((2, 2)),
    
    layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
    layers.BatchNormalization(),
    layers.MaxPooling2D((2, 2)),
    
    layers.Conv2D(256, (3, 3), activation='relu', padding='same'),
    layers.BatchNormalization(),
    layers.MaxPooling2D((2, 2)),
    
    layers.Conv2D(512, (3, 3), activation='relu', padding='same'),
    layers.BatchNormalization(),
    layers.MaxPooling2D((2, 2)),
    
    layers.Flatten(),
    layers.Dense(512, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(7, activation='softmax')
])

model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

model.summary()

lr_scheduler = ReduceLROnPlateau(
    monitor='val_loss',
    patience=3,
    factor=0.5,
    min_lr=1e-7
)

history = model.fit(
    train_generator,
    epochs=71,
    validation_data=test_generator,
    steps_per_epoch=len(train_generator),
    validation_steps=len(test_generator),
    callbacks=[lr_scheduler]
)

model.save("model1.h5")

final_train_acc = history.history["accuracy"][-1]
final_val_acc = history.history["val_accuracy"][-1]

print(f"Final Training Accuracy: {final_train_acc * 100:.2f}%")
print(f"Final Validation Accuracy: {final_val_acc * 100:.2f}%")