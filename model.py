import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from skimage import measure, morphology
from skimage.segmentation import watershed
from scipy import ndimage
import matplotlib.pyplot as plt
import numpy as np

class CellSegmentationTool:
    """
    A comprehensive tool for cell segmentation using an improved U-Net model
    with Batch Normalization and rigorous scientific metrics.
    """
    def __init__(self, input_shape=(256, 256, 3)):
        self.input_shape = input_shape
        self.model = None
        self.history = None

    def iou_coef(self, y_true, y_pred, smooth=1e-6):
        """Jaccard Index (Intersection over Union)."""
        y_true_f = tf.keras.backend.flatten(y_true)
        y_pred_f = tf.keras.backend.flatten(y_pred)
        intersection = tf.keras.backend.sum(y_true_f * y_pred_f)
        total = tf.keras.backend.sum(y_true_f) + tf.keras.backend.sum(y_pred_f)
        union = total - intersection
        return (intersection + smooth) / (union + smooth)

    def dice_coef(self, y_true, y_pred, smooth=1e-6):
        y_true_f = tf.keras.backend.flatten(y_true)
        y_pred_f = tf.keras.backend.flatten(y_pred)
        intersection = tf.keras.backend.sum(y_true_f * y_pred_f)
        return (2. * intersection + smooth) / (tf.keras.backend.sum(y_true_f) + tf.keras.backend.sum(y_pred_f) + smooth)

    def dice_loss(self, y_true, y_pred):
        return 1 - self.dice_coef(y_true, y_pred)

    def combined_loss(self, y_true, y_pred):
        bce = tf.keras.losses.binary_crossentropy(y_true, y_pred)
        dice = self.dice_loss(y_true, y_pred)
        return tf.keras.backend.mean(bce) + dice

    def _conv_block(self, inputs, filters, dropout_rate=0):
        """Helper: Conv -> BN -> ReLU -> Dropout -> Conv -> BN -> ReLU"""
        x = layers.Conv2D(filters, (3, 3), padding="same", kernel_initializer="he_normal", use_bias=False)(inputs)
        x = layers.BatchNormalization()(x)
        x = layers.Activation("relu")(x)
        
        if dropout_rate > 0:
            x = layers.Dropout(dropout_rate)(x)
            
        x = layers.Conv2D(filters, (3, 3), padding="same", kernel_initializer="he_normal", use_bias=False)(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation("relu")(x)
        return x

    def build_unet(self):
        inputs = keras.Input(shape=self.input_shape)

        # Encoder
        c1 = self._conv_block(inputs, 64, 0.1)
        p1 = layers.MaxPooling2D((2, 2))(c1)

        c2 = self._conv_block(p1, 128, 0.1)
        p2 = layers.MaxPooling2D((2, 2))(c2)

        c3 = self._conv_block(p2, 256, 0.2)
        p3 = layers.MaxPooling2D((2, 2))(c3)

        c4 = self._conv_block(p3, 512, 0.2)
        p4 = layers.MaxPooling2D(pool_size=(2, 2))(c4)

        # Bottleneck
        c5 = self._conv_block(p4, 1024, 0.3)

        # Decoder
        u6 = layers.Conv2DTranspose(512, (2, 2), strides=(2, 2), padding='same')(c5)
        u6 = layers.concatenate([u6, c4])
        c6 = self._conv_block(u6, 512, 0.2)

        u7 = layers.Conv2DTranspose(256, (2, 2), strides=(2, 2), padding='same')(c6)
        u7 = layers.concatenate([u7, c3])
        c7 = self._conv_block(u7, 256, 0.2)

        u8 = layers.Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(c7)
        u8 = layers.concatenate([u8, c2])
        c8 = self._conv_block(u8, 128, 0.1)

        u9 = layers.Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(c8)
        u9 = layers.concatenate([u9, c1], axis=3)
        c9 = self._conv_block(u9, 64, 0.1)

        outputs = layers.Conv2D(1, (1, 1), activation='sigmoid')(c9)

        self.model = keras.Model(inputs=[inputs], outputs=[outputs])
        return self.model

    def compile_model(self):
        if self.model is None:
            raise ValueError("Model must be built first.")
        self.model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=1e-4),
            loss=self.combined_loss,
            metrics=[self.dice_coef, self.iou_coef, 'binary_accuracy']
        )

    def train_model(self, train_gen, val_data, epochs=50, steps_per_epoch=None):
        """Supports both raw arrays and generators."""
        callbacks = [
            keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True, monitor='val_loss'),
            keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=5, monitor='val_loss'),
            keras.callbacks.ModelCheckpoint('best_model.keras', save_best_only=True, monitor='val_loss')
        ]
        
        # Check if input is a generator or a tuple of arrays
        if isinstance(train_gen, tuple):
             X_train, y_train = train_gen
             self.history = self.model.fit(
                X_train, y_train,
                validation_data=val_data,
                epochs=epochs,
                batch_size=8,
                callbacks=callbacks,
                verbose=1
            )
        else:
            self.history = self.model.fit(
                train_gen,
                validation_data=val_data,
                epochs=epochs,
                steps_per_epoch=steps_per_epoch,
                callbacks=callbacks,
                verbose=1
            )
        return self.history

    def predict_segmentation(self, image):
        if len(image.shape) == 3:
            image = np.expand_dims(image, axis=0)
        prediction = self.model.predict(image)
        return prediction[0, :, :, 0]

    def post_process_and_count(self, mask, threshold=0.5, min_size=50):
        binary_mask = (mask > threshold).astype(np.uint8)
        cleared_mask = morphology.remove_small_objects(binary_mask.astype(bool), min_size=min_size)
        distance = ndimage.distance_transform_edt(cleared_mask)
        coords = morphology.peak_local_max(distance, min_distance=10, labels=cleared_mask)
        markers_mask = np.zeros(distance.shape, dtype=bool)
        markers_mask[tuple(coords.T)] = True
        markers = measure.label(markers_mask)
        labels = watershed(-distance, markers, mask=cleared_mask)
        cell_count = len(np.unique(labels)) - 1
        return cell_count, labels
        
    def visualize_results(self, image, true_mask, raw_prediction, labeled_mask, cell_count):
        fig, axes = plt.subplots(1, 4, figsize=(20, 5))
        axes[0].imshow(image)
        axes[0].set_title('Original')
        axes[1].imshow(true_mask, cmap='gray')
        axes[1].set_title('Ground Truth')
        axes[2].imshow(raw_prediction, cmap='gray')
        axes[2].set_title('Prediction')
        colored_labels = plt.cm.nipy_spectral(labeled_mask / (labeled_mask.max() or 1))
        colored_labels[labeled_mask == 0] = 0
        axes[3].imshow(colored_labels)
        axes[3].set_title(f'Count: {cell_count}')
        for ax in axes: ax.axis('off')
        plt.tight_layout()
        plt.show()

    def evaluate_model(self, X_test, y_test):
        results = self.model.evaluate(X_test, y_test, verbose=0)
        return dict(zip(self.model.metrics_names, results))
