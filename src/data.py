import os
import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator

def load_dataset(dataset_path):
    image_paths = []
    labels = []
    class_folders = os.listdir(dataset_path)
    for label in class_folders:
        folder_path = os.path.join(dataset_path, label)
        for img_file in os.listdir(folder_path):
            image_paths.append(os.path.join(folder_path, img_file))
            labels.append(label)
    df = pd.DataFrame({
        "image_path": image_paths,
        "label": labels
    })
    return df

def split_dataset(df, test_size=0.15, val_size=0.18, random_state=42):
    train_val_df, test_df = train_test_split(df, test_size=test_size, stratify=df['label'], random_state=random_state)
    train_df, val_df = train_test_split(train_val_df, test_size=val_size, stratify=train_val_df['label'], random_state=random_state)
    return train_df, val_df, test_df

def create_generators(train_df, val_df, test_df, target_size=(224, 224), batch_size=32):
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        horizontal_flip=True,
        rotation_range=20,
        zoom_range=0.15,
        shear_range=0.15,
        fill_mode='nearest'
    )
    val_test_datagen = ImageDataGenerator(rescale=1./255)
    train_generator = train_datagen.flow_from_dataframe(
        dataframe=train_df,
        x_col='image_path',
        y_col='label',
        target_size=target_size,
        batch_size=batch_size,
        class_mode='categorical'
    )
    val_generator = val_test_datagen.flow_from_dataframe(
        dataframe=val_df,
        x_col='image_path',
        y_col='label',
        target_size=target_size,
        batch_size=batch_size,
        class_mode='categorical'
    )
    test_generator = val_test_datagen.flow_from_dataframe(
        dataframe=test_df,
        x_col='image_path',
        y_col='label',
        target_size=target_size,
        batch_size=batch_size,
        class_mode='categorical',
        shuffle=False
    )
    return train_generator, val_generator, test_generator