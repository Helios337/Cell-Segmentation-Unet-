"""
U-Net model for cell segmentation with post-processing and counting.
"""

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
    A comprehensive tool for cell segmentation using a U-Net architecture.

    Encapsulates model building, training, prediction, evaluation, and
    post-processing with watershed-based cell counting.

    Usage::

        tool = CellSegmentationTool(input_shape=(256, 256, 3))
        tool.build_unet()
        tool.compile_model()
        tool.train_model(X_train, y_train, X_val, y_val, epochs=25)
        metrics = tool.evaluate_model(X_test, y_test)
        mask = tool.predict_segmentation(image)
        count, labels = tool.post_process_and_count(mask)
    """

    def __init__(self, input_shape=(256, 256, 3)):
        self.input_shape = input_shape
        self.model = None
        self.history = None

    def build_unet(self):
        """Build the U-Net architecture for biomedical image segmentation."""
        inputs = keras.Input(shape=self.input_shape)

        # Encoder (contracting path)
        c1 = layers.Conv2D(64, (3, 3), activation='relu',
                           kernel_initializer='he_normal', padding='same')(inputs)
        c1 = layers.Dropout(0.1)(c1)
        c1 = layers.Conv2D(64, (3, 3), activation='relu',
                           kernel_initializer='he_normal', padding='same')(c1)
        p1 = layers.MaxPooling2D((2, 2))(c1)

        c2 = layers.Conv2D(128, (3, 3), activation='relu',
                           kernel_initializer='he_normal', padding='same')(p1)
        c2 = layers.Dropout(0.1)(c2)
        c2 = layers.Conv2D(128, (3, 3), activation='relu',
                           kernel_initializer='he_normal', padding='same')(c2)
        p2 = layers.MaxPooling2D((2, 2))(c2)

        c3 = layers.Conv2D(256, (3, 3), activation='relu',
                           kernel_initializer='he_normal', padding='same')(p2)
        c3 = layers.Dropout(0.2)(c3)
        c3 = layers.Conv2D(256, (3, 3), activation='relu',
                           kernel_initializer='he_normal', padding='same')(c3)
        p3 = layers.MaxPooling2D((2, 2))(c3)

        c4 = layers.Conv2D(512, (3, 3), activation='relu',
                           kernel_initializer='he_normal', padding='same')(p3)
        c4 = layers.Dropout(0.2)(c4)
        c4 = layers.Conv2D(512, (3, 3), activation='relu',
                           kernel_initializer='he_normal', padding='same')(c4)
        p4 = layers.MaxPooling2D(pool_size=(2, 2))(c4)

        # Bottleneck
        c5 = layers.Conv2D(1024, (3, 3), activation='relu',
                           kernel_initializer='he_normal', padding='same')(p4)
        c5 = layers.Dropout(0.3)(c5)
        c5 = layers.Conv2D(1024, (3, 3), activation='relu',
                           kernel_initializer='he_normal', padding='same')(c5)

        # Decoder (expanding path)
        u6 = layers.Conv2DTranspose(512, (2, 2), strides=(2, 2), padding='same')(c5)
        u6 = layers.concatenate([u6, c4])
        c6 = layers.Conv2D(512, (3, 3), activation='relu',
                           kernel_initializer='he_normal', padding='same')(u6)
        c6 = layers.Dropout(0.2)(c6)
        c6 = layers.Conv2D(512, (3, 3), activation='relu',
                           kernel_initializer='he_normal', padding='same')(c6)

        u7 = layers.Conv2DTranspose(256, (2, 2), strides=(2, 2), padding='same')(c6)
        u7 = layers.concatenate([u7, c3])
        c7 = layers.Conv2D(256, (3, 3), activation='relu',
                           kernel_initializer='he_normal', padding='same')(u7)
        c7 = layers.Dropout(0.2)(c7)
        c7 = layers.Conv2D(256, (3, 3), activation='relu',
                           kernel_initializer='he_normal', padding='same')(c7)

        u8 = layers.Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(c7)
        u8 = layers.concatenate([u8, c2])
        c8 = layers.Conv2D(128, (3, 3), activation='relu',
                           kernel_initializer='he_normal', padding='same')(u8)
        c8 = layers.Dropout(0.1)(c8)
        c8 = layers.Conv2D(128, (3, 3), activation='relu',
                           kernel_initializer='he_normal', padding='same')(c8)

        u9 = layers.Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(c8)
        u9 = layers.concatenate([u9, c1], axis=3)
        c9 = layers.Conv2D(64, (3, 3), activation='relu',
                           kernel_initializer='he_normal', padding='same')(u9)
        c9 = layers.Dropout(0.1)(c9)
        c9 = layers.Conv2D(64, (3, 3), activation='relu',
                           kernel_initializer='he_normal', padding='same')(c9)

        outputs = layers.Conv2D(1, (1, 1), activation='sigmoid')(c9)

        self.model = keras.Model(inputs=[inputs], outputs=[outputs])
        return self.model

    @staticmethod
    def dice_coef(y_true, y_pred, smooth=1e-6):
        """Dice coefficient for segmentation evaluation."""
        y_true_f = tf.reshape(y_true, [-1])
        y_pred_f = tf.reshape(y_pred, [-1])
        intersection = tf.reduce_sum(y_true_f * y_pred_f)
        return (2. * intersection + smooth) / (
            tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f) + smooth
        )

    @staticmethod
    def dice_loss(y_true, y_pred):
        """Dice loss = 1 - Dice coefficient."""
        return 1 - CellSegmentationTool.dice_coef(y_true, y_pred)

    @staticmethod
    def combined_loss(y_true, y_pred):
        """Combined Binary Cross-Entropy + Dice loss.

        Returns a scalar loss per sample, averaged over the batch.
        """
        bce = tf.keras.losses.binary_crossentropy(y_true, y_pred)
        dice = CellSegmentationTool.dice_loss(y_true, y_pred)
        return tf.reduce_mean(bce) + dice

    def compile_model(self):
        """Compile the U-Net model with Adam optimizer and combined loss."""
        if self.model is None:
            raise ValueError("Model must be built first using build_unet()")
        self.model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=1e-4),
            loss=self.combined_loss,
            metrics=[self.dice_coef, 'binary_accuracy'],
        )
        return self.model

    def train_model(self, X_train, y_train, X_val, y_val,
                    epochs=50, batch_size=8):
        """Train the U-Net model with early stopping and checkpointing."""
        if self.model is None:
            raise ValueError("Model must be built and compiled first")

        callbacks = [
            keras.callbacks.EarlyStopping(
                patience=10, restore_best_weights=True, monitor='val_loss'
            ),
            keras.callbacks.ReduceLROnPlateau(
                factor=0.5, patience=5, monitor='val_loss'
            ),
            keras.callbacks.ModelCheckpoint(
                'best_model.h5', save_best_only=True, monitor='val_loss'
            ),
        ]
        self.history = self.model.fit(
            X_train, y_train,
            batch_size=batch_size,
            epochs=epochs,
            validation_data=(X_val, y_val),
            callbacks=callbacks,
            verbose=1,
        )
        return self.history

    def predict_segmentation(self, image):
        """Predict the raw segmentation probability map for a single image."""
        if self.model is None:
            raise ValueError("Model has not been trained or loaded.")
        if len(image.shape) == 3:
            image = np.expand_dims(image, axis=0)
        prediction = self.model.predict(image, verbose=0)
        return prediction[0, :, :, 0]

    @staticmethod
    def post_process_and_count(mask, threshold=0.5, min_size=50):
        """Post-process a raw probability mask and return cell count + labels.

        Uses morphological cleaning, distance transform, and watershed
        to separate touching cells.

        Args:
            mask: 2D float array, raw prediction (values in [0, 1]).
            threshold: Binarization threshold.
            min_size: Minimum cell size in pixels (smaller objects removed).

        Returns:
            Tuple of (cell_count, labeled_mask).
        """
        binary_mask = (mask > threshold).astype(np.uint8)
        cleared_mask = morphology.remove_small_objects(
            binary_mask.astype(bool), max_size=min_size
        )
        distance = ndimage.distance_transform_edt(cleared_mask)
        coords = morphology.local_maxima(distance)
        markers = measure.label(coords)
        labels = watershed(-distance, markers, mask=cleared_mask)
        cell_count = len(np.unique(labels)) - 1  # Subtract background
        return cell_count, labels

    @staticmethod
    def visualize_results(image, true_mask, raw_prediction,
                          labeled_mask, cell_count):
        """Display a 4-panel comparison: Original, Ground Truth,
        Raw Prediction, and Final Labeled Segmentation."""
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
        colored_labels = plt.cm.nipy_spectral(
            labeled_mask / (labeled_mask.max() or 1)
        )
        colored_labels[labeled_mask == 0] = 0
        axes[3].imshow(colored_labels)
        axes[3].set_title(f'Segmented Cells  Count: {cell_count}')
        axes[3].axis('off')
        plt.tight_layout()
        plt.show()

    def evaluate_model(self, X_test, y_test):
        """Evaluate model performance on the test set.

        Returns:
            Dict with keys 'loss', 'dice_coefficient', 'binary_accuracy'.
        """
        results = self.model.evaluate(X_test, y_test, verbose=0)
        return {
            'loss': results[0],
            'dice_coefficient': results[1],
            'binary_accuracy': results[2],
        }