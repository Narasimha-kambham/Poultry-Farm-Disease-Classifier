import os
from data import load_dataset, split_dataset, create_generators
from model import build_model
from utils import evaluate_model

def train(dataset_path, model_save_path='poultry_disease_model.keras', epochs=10, batch_size=32):
    df = load_dataset(dataset_path)
    train_df, val_df, test_df = split_dataset(df)
    train_gen, val_gen, test_gen = create_generators(train_df, val_df, test_df, batch_size=batch_size)
    model = build_model(num_classes=len(train_gen.class_indices))
    history = model.fit(
        train_gen,
        steps_per_epoch=train_gen.samples // batch_size,
        validation_data=val_gen,
        validation_steps=val_gen.samples // batch_size,
        epochs=epochs
    )
    model.save(model_save_path)
    print(f"Model saved to {model_save_path}")
    evaluate_model(model, test_gen)

if __name__ == "__main__":
    # Example usage: adjust the path as needed
    dataset_path = 'poultry_diseases'  # Assuming relative to project root
    train(dataset_path)