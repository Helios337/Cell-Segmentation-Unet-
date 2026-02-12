from tensorflow import keras
from tensorflow.keras import layers

def build_deep_unet(input_shape):
    """Builds a 5-level U-Net model."""
    inputs = keras.Input(shape=input_shape)
    
    def conv_block(x, filters, dropout=0.1):
        x = layers.Conv2D(filters, 3, padding="same", kernel_initializer="he_normal", use_bias=False)(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation("relu")(x)
        if dropout: x = layers.Dropout(dropout)(x)
        x = layers.Conv2D(filters, 3, padding="same", kernel_initializer="he_normal", use_bias=False)(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation("relu")(x)
        return x

    # Encoder
    c1 = conv_block(inputs, 32); p1 = layers.MaxPooling2D()(c1)
    c2 = conv_block(p1, 64); p2 = layers.MaxPooling2D()(c2)
    c3 = conv_block(p2, 128); p3 = layers.MaxPooling2D()(c3)
    c4 = conv_block(p3, 256); p4 = layers.MaxPooling2D()(c4)
    
    # Bottleneck
    c5 = conv_block(p4, 512, dropout=0.3)
    
    # Decoder
    u6 = layers.Conv2DTranspose(256, (2, 2), strides=(2, 2), padding='same')(c5)
    u6 = layers.concatenate([u6, c4])
    c6 = conv_block(u6, 256)
    
    u7 = layers.Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(c6)
    u7 = layers.concatenate([u7, c3])
    c7 = conv_block(u7, 128)
    
    u8 = layers.Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(c7)
    u8 = layers.concatenate([u8, c2])
    c8 = conv_block(u8, 64)
    
    u9 = layers.Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same')(c8)
    u9 = layers.concatenate([u9, c1])
    c9 = conv_block(u9, 32)
    
    outputs = layers.Conv2D(1, (1, 1), activation='sigmoid')(c9)
    
    model = keras.Model(inputs, outputs, name="Deep_UNet")
    model.compile(optimizer=keras.optimizers.Adam(1e-4), 
                  loss='binary_crossentropy', 
                  metrics=['accuracy', keras.metrics.BinaryIoU(target_class_ids=[1], name='iou')])
    return model
