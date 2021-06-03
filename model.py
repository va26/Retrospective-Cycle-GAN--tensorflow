import tensorflow as tf

assert tf.__version__.startswith('2')


class InstanceNormalization(tf.keras.layers.Layer):
    """Instance Normalization Layer (https://arxiv.org/abs/1607.08022)."""

    def __init__(self, epsilon=1e-5):
        super(InstanceNormalization, self).__init__()
        self.epsilon = epsilon

    def build(self, input_shape):
        self.scale = self.add_weight(
            name='scale',
            shape=input_shape[-1:],
            initializer=tf.random_normal_initializer(1., 0.02),
            trainable=True)

        self.offset = self.add_weight(
            name='offset',
            shape=input_shape[-1:],
            initializer='zeros',
            trainable=True)

    def call(self, x):
        mean, variance = tf.nn.moments(x, axes=[1, 2], keepdims=True)
        inv = tf.math.rsqrt(variance + self.epsilon)
        normalized = (x - mean) * inv
        return self.scale * normalized + self.offset


def first_convnet(filters, strides, size, norm_type='instancenorm', apply_norm=True, relu='relu', apply_relu=True):
    """Downsamples an input.

    Conv2D => Batchnorm => LeakyRelu

    Args:
      filters: number of filters
      size: filter size
      norm_type: Normalization type; either 'batchnorm' or 'instancenorm'.
      apply_norm: If True, adds the batchnorm layer

    Returns:
      Downsample Sequential Model
    """
    initializer = tf.random_normal_initializer(0., 0.02)

    result = tf.keras.Sequential()
    result.add(tf.keras.layers.Conv3D(filters, size, strides, padding='same',
                                      kernel_initializer=initializer, use_bias=False,
                                      input_shape=(None, None, None, 3)))

    if apply_norm:
        if norm_type.lower() == 'batchnorm':
            result.add(tf.keras.layers.BatchNormalization())
        elif norm_type.lower() == 'instancenorm':
            result.add(InstanceNormalization())

    if apply_relu:
        if relu == 'relu':
            result.add(tf.keras.layers.ReLU())
        elif relu == 'leakyrelu':
            result.add(tf.keras.layers.LeakyReLU(alpha=0.2))

    return result


def convnet(filters, strides, size, norm_type='instancenorm', apply_norm=True, relu='relu', apply_relu=True):
    """Downsamples an input.

    Conv2D => Batchnorm => LeakyRelu

    Args:
      filters: number of filters
      size: filter size
      norm_type: Normalization type; either 'batchnorm' or 'instancenorm'.
      apply_norm: If True, adds the `norm_type` layer

    Returns:
      Downsample Sequential Model
    """
    initializer = tf.random_normal_initializer(0., 0.02)

    result = tf.keras.Sequential()
    result.add(tf.keras.layers.Conv3D(filters, size, strides, padding='same', kernel_initializer=initializer,
                                      use_bias=False))

    if apply_norm:
        if norm_type.lower() == 'batchnorm':
            result.add(tf.keras.layers.BatchNormalization())
        elif norm_type.lower() == 'instancenorm':
            result.add(InstanceNormalization())

    if apply_relu:
        if relu == 'relu':
            result.add(tf.keras.layers.ReLU())
        elif relu == 'leakyrelu':
            result.add(tf.keras.layers.LeakyReLU(alpha=0.2))

    return result


def residual_block(filters, strides, size):
    """Downsamples an input.

    Conv2D => Batchnorm => LeakyRelu

    Args:
      filters: number of filters
      size: filter size

    Returns:
      Downsample Sequential Model
    """
    initializer = tf.random_normal_initializer(0., 0.02)

    result = tf.keras.Sequential()
    result.add(tf.keras.layers.Conv3D(filters, size, strides, padding='same',
                                      kernel_initializer=initializer, use_bias=False))

    result.add(InstanceNormalization())
    result.add(tf.keras.layers.ReLU())
    result.add(
        tf.keras.layers.Conv3D(filters, size, strides, padding='same',
                               kernel_initializer=initializer, use_bias=False))
    result.add(InstanceNormalization())

    return result


def convnet_transpose(filters, size, norm_type='instancenorm', apply_dropout=False, relu='relu', apply_relu=True):
    """Upsamples an input.

    Conv2DTranspose => Batchnorm => Dropout => Relu

    Args:
      filters: number of filters
      size: filter size
      norm_type: Normalization type; either 'batchnorm' or 'instancenorm'.
      apply_dropout: If True, adds the dropout layer

    Returns:
      Upsample Sequential Model
    """

    initializer = tf.random_normal_initializer(0., 0.02)

    result = tf.keras.Sequential()
    result.add(tf.keras.layers.Conv3DTranspose(filters, size, strides=(2, 2, 1), padding='same',
                                               kernel_initializer=initializer, use_bias=False))

    if norm_type.lower() == 'batchnorm':
        result.add(tf.keras.layers.BatchNormalization())
    elif norm_type.lower() == 'instancenorm':
        result.add(InstanceNormalization())

    if apply_dropout:
        result.add(tf.keras.layers.Dropout(0.5))

    if apply_relu:
        if relu == 'relu':
            result.add(tf.keras.layers.ReLU())
        elif relu == 'leakyrelu':
            result.add(tf.keras.layers.LeakyReLU(alpha=0.2))

    return result


def generator(output_channels, norm_type='instancenorm'):
    """
    Args:
      output_channels: Output channels
      norm_type: Type of normalization. Either 'batchnorm' or 'instancenorm'.

    Returns:
      Generator model
    """

    convnets = [first_convnet(128, (1, 1, 1), (7, 7, 4), norm_type, apply_norm=False, relu='relu', apply_relu=False),
                # (bs, 128, 128, 64)
                convnet(128, 2, (3, 3, 2), norm_type),  # (bs, 64, 64, 128)
                convnet(256, 2, (3, 3, 1), norm_type),  # (bs, 32, 32, 256)
                ]
    resnets = [residual_block(256, 1, (3, 3, 1)),
               residual_block(256, 1, (3, 3, 1)),
               residual_block(256, 1, (3, 3, 1)),
               residual_block(256, 1, (3, 3, 1)),
               residual_block(256, 1, (3, 3, 1)),
               residual_block(256, 1, (3, 3, 1)),
               residual_block(256, 1, (3, 3, 1)),
               residual_block(256, 1, (3, 3, 1)),
               residual_block(256, 1, (3, 3, 1))
               ]
    transpose_convnets = [convnet_transpose(128, (3, 3, 1), norm_type),  # (bs, 64, 64, 256)
                          convnet_transpose(256, (3, 3, 1), norm_type),  # (bs, 128, 128, 128)
                          ]

    initializer = tf.random_normal_initializer(0., 0.02)
    last = tf.keras.layers.Conv3D(output_channels, (7, 7, 1), strides=1,
                                  padding='same', kernel_initializer=initializer,
                                  activation='tanh')  # (bs, 256, 256, 3)

    add = tf.keras.layers.Add()

    inputs = tf.keras.layers.Input(shape=[None, None, None, 3])
    x = inputs

    # Downsampling through the model
    for conv in convnets:
        x = conv(x)

    for res in resnets:
        skip = x
        x = res(x)
        x = add([x, skip])

    # Upsampling and establishing the skip connections
    for convT in transpose_convnets:
        x = convT(x)

    x = last(x)

    return tf.keras.Model(inputs=inputs, outputs=x)


def discriminator(norm_type='instancenorm', target=True):
    """
    Args:
      norm_type: Type of normalization. Either 'batchnorm' or 'instancenorm'.
      target: Bool, indicating whether target image is an input or not.

    Returns:
      Discriminator model
    """

    initializer = tf.random_normal_initializer(0., 0.02)

    inp = tf.keras.layers.Input(shape=[None, None, None, 3], name='input_image')
    x = inp

    if target:
        tar = tf.keras.layers.Input(shape=[None, None, None, 3], name='target_image')
        x = tf.keras.layers.concatenate([inp, tar])  # (bs, 256, 256, channels*2)

    conv1 = convnet(64, (2, 2, 1), (1, 4, 4), norm_type, apply_norm=False, relu='leakyrelu')(x)  # (bs, 128, 128, 64)
    conv2 = convnet(128, (2, 2, 1), (1, 4, 4), norm_type, relu='leakyrelu')(conv1)  # (bs, 64, 64, 128)
    conv3 = convnet(256, (2, 2, 1), (1, 4, 4), norm_type, relu='leakyrelu')(conv2)  # (bs, 32, 32, 256)
    conv4 = convnet(512, 1, (1, 4, 4), norm_type, relu='leakyrelu')(conv3)

    last = tf.keras.layers.Conv3D(1, (4, 4, 1), strides=1, kernel_initializer=initializer)(conv4)  # (bs, 30, 30, 1)

    if target:
        return tf.keras.Model(inputs=[inp, tar], outputs=last)
    else:
        return tf.keras.Model(inputs=inp, outputs=last)


def discriminator_seq(norm_type='instancenorm', target=True):
    """
    Args:
      norm_type: Type of normalization. Either 'batchnorm' or 'instancenorm'.
      target: Bool, indicating whether target image is an input or not.

    Returns:
      Discriminator model
    """

    initializer = tf.random_normal_initializer(0., 0.02)

    inp = tf.keras.layers.Input(shape=[None, None, None, 3], name='input_image')
    x = inp

    if target:
        tar = tf.keras.layers.Input(shape=[None, None, None, 3], name='target_image')
        x = tf.keras.layers.concatenate([inp, tar])  # (bs, 256, 256, channels*2)

    conv1 = first_convnet(64, 2, 4, 'instancenorm', apply_norm=False, relu='leakyrelu')(x)  # (bs, 128, 128, 64)
    conv2 = convnet(128, 2, 4, norm_type, relu='leakyrelu')(conv1)  # (bs, 64, 64, 128)
    conv3 = convnet(256, 2, 4, norm_type, relu='leakyrelu')(conv2)  # (bs, 32, 32, 256)
    conv4 = convnet(512, 1, 4, norm_type, relu='leakyrelu')(conv3)

    last = tf.keras.layers.Conv3D(1, (4, 4, 1), strides=1, kernel_initializer=initializer)(conv4)  # (bs, 30, 30, 1)

    if target:
        return tf.keras.Model(inputs=[inp, tar], outputs=last)
    else:
        return tf.keras.Model(inputs=inp, outputs=last)
