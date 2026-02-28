import os
import argparse
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical
from dataset import create_dataset_dataframe, load_data_to_memory
from model import build_hybrid_model
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau

def train_model(data_dir, epochs=50, batch_size=32):
    """
    Full training pipeline for SER model.
    """
    print(f"Loading data from {data_dir}...")
    df = create_dataset_dataframe(data_dir)
    
    if df.empty:
        print("No data found! Please check the dataset path.")
        return
        
    print(f"Found {len(df)} files. Extracting features... (This might take a while)")
    X, y = load_data_to_memory(df, augment=True) # enabling augmentation
    
    if len(X) == 0:
        print("Feature extraction failed.")
        return

    print("Encoding labels...")
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    y_categorical = to_categorical(y_encoded)
    np.save('models/label_encoder_classes.npy', le.classes_)
    
    # Reshape X for CNN + LSTM (samples, timesteps, features)
    # The features vector (e.g., shape 162) can be reshaped to treat as a sequence,
    # or ideally replaced with actual 2D Mel Spectrograms. 
    # For now, reshaping 1D features into (features, 1) sequence for simplicity.
    X_reshaped = np.expand_dims(X, axis=-1) 

    # Split dataset
    X_train, X_test, y_train, y_test = train_test_split(X_reshaped, y_categorical, test_size=0.2, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)
    
    print(f"Train Shape: {X_train.shape}, Val Shape: {X_val.shape}, Test Shape: {X_test.shape}")
    
    # Build Model
    input_shape = (X_train.shape[1], X_train.shape[2])
    num_classes = y_train.shape[1]
    model = build_hybrid_model(input_shape=input_shape, num_classes=num_classes)
    
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
                  
    # Callbacks
    checkpoint = ModelCheckpoint('models/best_model.h5', monitor='val_accuracy', save_best_only=True, verbose=1)
    early_stop = EarlyStopping(monitor='val_accuracy', patience=10, verbose=1, restore_best_weights=True)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, verbose=1)
    
    print("Starting Training...")
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=[checkpoint, early_stop, reduce_lr]
    )
    
    print("Evaluating Model on Test Data...")
    loss, accuracy = model.evaluate(X_test, y_test)
    print(f"Test Accuracy: {accuracy * 100:.2f}%")
    
    model.save('models/final_model.h5')
    print("Model saved to models/final_model.h5")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train SER Hybrid Model")
    parser.add_argument("--data_dir", type=str, default="data/", help="Path to RAVDESS/TESS datasets")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=32)
    args = parser.parse_args()
    
    train_model(args.data_dir, args.epochs, args.batch_size)
