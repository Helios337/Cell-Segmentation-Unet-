import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
from tensorflow import keras

from src.data_loader import RealBiologicalLoader
from src.model import build_deep_unet

# Configuration
IMG_SIZE = (128, 128)
BATCH_SIZE = 16
EPOCHS = 20
SEED = 42

def combined_generator(gen1, gen2):
    """Yields tuples of (img, mask) indefinitely."""
    while True:
        yield (next(gen1), next(gen2))

def main():
    # 1. Load Data
    loader = RealBiologicalLoader()
    X, y = loader.load_dataset(img_size=IMG_SIZE)
    
    if X is None:
        print("Dataset loading failed.")
        return

    # 2. Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=SEED)
    
    # Save test set for evaluation script
    np.save("X_test.npy", X_test)
    np.save("y_test.npy", y_test)
    print("Test set saved for evaluation.")

    # 3. Augmentation
    data_gen_args = dict(rotation_range=90, width_shift_range=0.1, height_shift_range=0.1,
                         zoom_range=0.2, horizontal_flip=True, vertical_flip=True, fill_mode='reflect')
    
    image_datagen = ImageDataGenerator(**data_gen_args)
    mask_datagen = ImageDataGenerator(**data_gen_args)
    
    train_gen = combined_generator(
        image_datagen.flow(X_train, batch_size=BATCH_SIZE, seed=SEED),
        mask_datagen.flow(y_train, batch_size=BATCH_SIZE, seed=SEED)
    )

    # 4. Train
    model = build_deep_unet(input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3))
    
    callbacks = [
        keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True, monitor='val_loss'),
        keras.callbacks.ModelCheckpoint('best_model.keras', save_best_only=True)
    ]
    
    print("Starting Training...")
    model.fit(
        train_gen,
        steps_per_epoch=len(X_train) // BATCH_SIZE,
        validation_data=(X_test, y_test),
        epochs=EPOCHS,
        callbacks=callbacks
    )
    print("Training Complete. Model saved as 'best_model.keras'")

if __name__ == "__main__":
    main()
