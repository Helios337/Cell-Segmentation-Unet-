from tensorflow import keras
from tensorflow.keras import layers


def build_deep_unet(
    input_shape,
    base_filters=32,
    depth=4,
    dropout=0.1,
    bottleneck_dropout=0.3,
    learning_rate=1e-4,
    use_separable=False,
):
    """Builds a configurable U-Net model."""
    inputs = keras.Input(shape=input_shape)

    def create_conv_layer(filters):
        if use_separable:
            return layers.SeparableConv2D(
                filters,
                3,
                padding="same",
                depthwise_initializer="he_normal",
                pointwise_initializer="he_normal",
                use_bias=False,
            )
        return layers.Conv2D(
            filters,
            3,
            padding="same",
            kernel_initializer="he_normal",
            use_bias=False,
        )

    def conv_block(x, filters, block_dropout=0.1):
        x = create_conv_layer(filters)(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation("relu")(x)
        if block_dropout:
            x = layers.Dropout(block_dropout)(x)
        x = create_conv_layer(filters)(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation("relu")(x)
        return x

    skips = []
    x = inputs

    for i in range(depth):
        filters = base_filters * (2**i)
        c = conv_block(x, filters, block_dropout=dropout)
        skips.append(c)
        x = layers.MaxPooling2D()(c)

    bottleneck_filters = base_filters * (2**depth)
    x = conv_block(x, bottleneck_filters, block_dropout=bottleneck_dropout)

    for i in reversed(range(depth)):
        filters = base_filters * (2**i)
        x = layers.Conv2DTranspose(filters, (2, 2), strides=(2, 2), padding="same")(x)
        x = layers.concatenate([x, skips[i]])
        x = conv_block(x, filters, block_dropout=dropout)

    outputs = layers.Conv2D(1, (1, 1), activation="sigmoid")(x)

    model = keras.Model(inputs, outputs, name="Deep_UNet")
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate),
        loss="binary_crossentropy",
        metrics=["accuracy", keras.metrics.BinaryIoU(target_class_ids=[1], name="iou")],
    )
    return model
