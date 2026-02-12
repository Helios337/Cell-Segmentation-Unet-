def build_unet(self):
        inputs = keras.Input(shape=self.input_shape)

        def encoder_block(input_tensor, num_filters, dropout_rate=0.0):
            x = layers.Conv2D(num_filters, (3, 3), padding="same", kernel_initializer="he_normal")(input_tensor)
            x = layers.BatchNormalization()(x) # Added Batch Norm
            x = layers.Activation("relu")(x)
            
            if dropout_rate > 0:
                x = layers.Dropout(dropout_rate)(x)
            
            x = layers.Conv2D(num_filters, (3, 3), padding="same", kernel_initializer="he_normal")(x)
            x = layers.BatchNormalization()(x) # Added Batch Norm
            x = layers.Activation("relu")(x)
            
            p = layers.MaxPooling2D((2, 2))(x)
            return x, p

        def decoder_block(input_tensor, concat_tensor, num_filters, dropout_rate=0.0):
            x = layers.Conv2DTranspose(num_filters, (2, 2), strides=(2, 2), padding="same")(input_tensor)
            x = layers.concatenate([x, concat_tensor])
            
            x = layers.Conv2D(num_filters, (3, 3), padding="same", kernel_initializer="he_normal")(x)
            x = layers.BatchNormalization()(x)
            x = layers.Activation("relu")(x)
            
            if dropout_rate > 0:
                x = layers.Dropout(dropout_rate)(x)

            x = layers.Conv2D(num_filters, (3, 3), padding="same", kernel_initializer="he_normal")(x)
            x = layers.BatchNormalization()(x)
            x = layers.Activation("relu")(x)
            return x

        # Encoder
        c1, p1 = encoder_block(inputs, 64, 0.1)
        c2, p2 = encoder_block(p1, 128, 0.1)
        c3, p3 = encoder_block(p2, 256, 0.2)
        c4, p4 = encoder_block(p3, 512, 0.2)

        # Bottleneck
        b1 = layers.Conv2D(1024, (3, 3), padding="same", kernel_initializer="he_normal")(p4)
        b1 = layers.BatchNormalization()(b1)
        b1 = layers.Activation("relu")(b1)
        b1 = layers.Dropout(0.3)(b1)
        b1 = layers.Conv2D(1024, (3, 3), padding="same", kernel_initializer="he_normal")(b1)
        b1 = layers.BatchNormalization()(b1)
        b1 = layers.Activation("relu")(b1)

        # Decoder
        u6 = decoder_block(b1, c4, 512, 0.2)
        u7 = decoder_block(u6, c3, 256, 0.2)
        u8 = decoder_block(u7, c2, 128, 0.1)
        u9 = decoder_block(u8, c1, 64, 0.1)

        outputs = layers.Conv2D(1, (1, 1), activation='sigmoid')(u9)

        self.model = keras.Model(inputs=[inputs], outputs=[outputs])
        return self.model
