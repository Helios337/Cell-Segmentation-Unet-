import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from skimage import measure, morphology
from skimage.segmentation import watershed
from scipy import ndimage
import matplotlib.pyplot as plt

class CellSegmentationTool:
    """
    A comprehensive tool for cell segmentation using a U-Net model.
    Encapsulates model building, training, prediction, and evaluation.
    """
    def __init__(self, input_shape=(256, 256, 3)):
        """Initializes the tool with the specified input shape for images."""
        self.input_shape = input_shape
        self.model = None
        self.history = None

    def build_unet(self):
        """Builds the U-Net architecture for cell segmentation."""
        inputs = keras.Input(shape=self.input_shape)

        # Encoder (Contracting Path)
        c1 = layers.Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(inputs)
        c1 = layers.Dropout(0.1)(c1)
        c1 = layers.Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c1)
        p1 = layers.MaxPooling2D((2, 2))(c1)

        c2 = layers.Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p1)
        c2 = layers.Dropout(0.1)(c2)
        c2 = layers.Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c2)
        p2 = layers.MaxPooling2D((2, 2))(c2)

        c3 = layers.Conv2D(256, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p2)
        c3 = layers.Dropout(0.2)(c3)
        c3 = layers.Conv2D(256, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c3)
        p3 = layers.MaxPooling2D((2, 2))(c3)

        c4 = layers.Conv2D(512, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p3)
        c4 = layers.Dropout(0.2)(c4)
        c4 = layers.Conv2D(512, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c4)
        p4 = layers.MaxPooling2D(pool_size=(2, 2))(c4)

        # Bottleneck
        c5 = layers.Conv2D(1024, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p4)
        c5 = layers.Dropout(0.3)(c5)
        c5 = layers.Conv2D(1024, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c5)

        # Decoder (Expanding Path)
        u6 = layers.Conv2DTranspose(512, (2, 2), strides=(2, 2), padding='same')(c5)
        u6 = layers.concatenate([u6, c4])
        c6 = layers.Conv2D(512, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u6)
        c6 = layers.Dropout(0.2)(c6)
        c6 = layers.Conv2D(512, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c6)

        u7 = layers.Conv2DTranspose(256, (2, 2), strides=(2, 2), padding='same')(c6)
        u7 = layers.concatenate([u7, c3])
        c7 = layers.Conv2D(256, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u7)
        c7 = layers.Dropout(0.2)(c7)
        c7 = layers.Conv2D(256, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c7)

        u8 = layers.Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(c7)
        u8 = layers.concatenate([u8, c2])
        c8 = layers.Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u8)
        c8 = layers.Dropout(0.1)(c8)
        c8 = layers.Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c8)

        u9 = layers.Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(c8)
        u9 = layers.concatenate([u9, c1], axis=3)
        c9 = layers.Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u9)
        c9 = layers.Dropout(0.1)(c9)
        c9 = layers.Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c9)

        outputs = layers.Conv2D(1, (1, 1), activation='sigmoid')(c9)

        self.model = keras.Model(inputs=[inputs], outputs=[outputs])
        return self.model

    def dice_coef(self, y_true, y_pred, smooth=1e-6):
        """Dice coefficient for segmentation evaluation."""
        y_true_f = tf.keras.backend.flatten(y_true)
        y_pred_f = tf.keras.backend.flatten(y_pred)
        intersection = tf.keras.backend.sum(y_true_f * y_pred_f)
        return (2. * intersection + smooth) / (tf.keras.backend.sum(y_true_f) + tf.keras.backend.sum(y_pred_f) + smooth)

    def dice_loss(self, y_true, y_pred):
        """Dice loss function."""
        return 1 - self.dice_coef(y_true, y_pred)

    def combined_loss(self, y_true, y_pred):
        """Combined Binary Cross Entropy + Dice Loss."""
        bce = tf.keras.losses.binary_crossentropy(y_true, y_pred)
        dice = self.dice_loss(y_true, y_pred)
        return tf.keras.backend.mean(bce) + dice

    def compile_model(self):
        """Compiles the U-Net model with optimizer, loss, and metrics."""
        if self.model is None:
            raise ValueError("Model must be built first using build_unet()")
        self.model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=1e-4),
            loss=self.combined_loss,
            metrics=[self.dice_coef, 'binary_accuracy']
        )
        return self.model

    def train_model(self, X_train, y_train, X_val, y_val, epochs=50, batch_size=8):
        """Trains the U-Net model."""
        if self.model is None:
            raise ValueError("Model must be built and compiled first")
        callbacks = [
            keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True, monitor='val_loss'),
            keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=5, monitor='val_loss'),
            keras.callbacks.ModelCheckpoint('best_model.h5', save_best_only=True, monitor='val_loss')
        ]
        self.history = self.model.fit(
            X_train, y_train,
            batch_size=batch_size,
            epochs=epochs,
            validation_data=(X_val, y_val),
            callbacks=callbacks,
            verbose=1
        )
        return self.history

    def predict_segmentation(self, image):
        """
        Predicts the raw segmentation mask for a single image.
        This method now correctly returns only the raw probability map.
        """
        if self.model is None:
            raise ValueError("Model has not been trained or loaded.")
        if len(image.shape) == 3:
            image = np.expand_dims(image, axis=0)
        prediction = self.model.predict(image)
        return prediction[0, :, :, 0]

    def post_process_and_count(self, mask, threshold=0.5, min_size=50):
        """
        This new method contains the post-processing and counting logic.
        It takes a raw prediction mask as input.
        """
        binary_mask = (mask > threshold).astype(np.uint8)
        cleared_mask = morphology.remove_small_objects(binary_mask.astype(bool), min_size=min_size)
        distance = ndimage.distance_transform_edt(cleared_mask)
        
        # Identify local maxima
        coords = morphology.peak_local_max(distance, min_distance=10, labels=cleared_mask)
        markers_mask = np.zeros(distance.shape, dtype=bool)
        markers_mask[tuple(coords.T)] = True
        
        markers = measure.label(markers_mask)
        labels = watershed(-distance, markers, mask=cleared_mask)
        cell_count = len(np.unique(labels)) - 1
        return cell_count, labels

    def visualize_results(self, image, true_mask, raw_prediction, labeled_mask, cell_count):
        """
        Updated to visualize a 4-panel comparison: Original, Ground Truth,
        Raw Prediction, and Final Labeled Segmentation with the count.
        """
        fig, axes = plt.subplots(1, 4, figsize=(20, 5))
        axes[0].imshow(image)
        axes[0].set_title('Original Image')
        axes[0].axis('off')
        axes[1].imshow(true_mask, cmap='gray')
        axes[1].set_title('Ground Truth Mask')
        axes[1].axis('off')
        axes[2].imshow(raw_prediction, cmap='gray')
        axes[2].set_title('Raw U-Net Prediction')
        axes[2].axis('off')
        
        colored_labels = plt.cm.nipy_spectral(labeled_mask / (labeled_mask.max() or 1))
        colored_labels[labeled_mask == 0] = 0
        axes[3].imshow(colored_labels)
        axes[3].set_title(f'Segmented Cells\nCount: {cell_count}')
        axes[3].axis('off')
        plt.tight_layout()
        plt.show()

    def evaluate_model(self, X_test, y_test):
        """Evaluates the model's performance on the test set."""
        results = self.model.evaluate(X_test, y_test, verbose=0)
        metrics_dict = {
            'loss': results[0],
            'dice_coefficient': results[1],
            'binary_accuracy': results[2]
        }
        return metrics_dict
