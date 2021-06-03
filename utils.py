import logging
import tensorflow as tf
import math
import pickle as pkl


def setup_logger(name, log_file, level=logging.INFO):
    """
    To setup as many loggers as you want
    :rtype:
    """
    handler = logging.FileHandler(log_file)
    formatter = logging.Formatter('%(asctime)s %(levelname)s %(message)s')
    handler.setFormatter(formatter)

    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.addHandler(handler)

    return logger


def load_data(dfile, GLOBAL_BATCH_SIZE, strategy):
    """
    Loads pickle data from the source folder and returns a distributed dataset for parallel training on GPUs
    :rtype: tf.distribute.DistributedDataset, could be thought of as a "distributed" dataset
    """
    with open(dfile, 'rb') as f:
        img_data = pkl.load(f)

    dataset = tf.data.Dataset.from_tensor_slices(img_data)
    train_dataset = dataset.batch(GLOBAL_BATCH_SIZE, drop_remainder=True)
    train_dist_dataset = strategy.experimental_distribute_dataset(train_dataset)

    return train_dist_dataset


def laplacian_of_gaussian(image, filter_size, sigma):
    """
    The function runs the laplacian of gaussian filter using the filter_size and sigma
    provided by the user over imput 'image'
    :rtype:
    """
    image = tf.squeeze(image, 3)
    if len(image.shape) == 4 and image.shape[3] == 3:  # convert rgb to grayscale
        image = tf.image.convert_image_dtype(image, tf.dtypes.float32)
        image = tf.image.rgb_to_grayscale(image)
        image = tf.image.convert_image_dtype(image, tf.dtypes.float32)
        image = tf.squeeze(image, 3)

    elif len(image.shape) > 3:
        raise TypeError('Incorrect number of channels.')
    n_channels = 1
    image = tf.expand_dims(image, 3)

    w = math.ceil(sigma * filter_size)
    w_range = int(math.floor(w / 2))

    y = x = tf.range(-w_range, w_range + 1, 1)
    Y, X = tf.meshgrid(x, y)
    z = tf.cast(tf.add(tf.square(X), tf.square(Y)), tf.float32)
    nom = tf.subtract(z, 2 * (sigma ** 2))
    denom = 2 * math.pi * (sigma ** 6)
    exp = tf.exp(-z / 2 * (sigma ** 2))
    fil = tf.divide(tf.multiply(nom, exp), denom)

    fil = tf.stack([fil] * n_channels, axis=2)
    fil = tf.expand_dims(fil, 3)

    new = tf.image.convert_image_dtype(image, tf.dtypes.float32)
    res = tf.nn.depthwise_conv2d(new, fil, strides=[1, 1, 1, 1], padding="SAME")
    res = tf.squeeze(res, 3)

    minM = tf.math.reduce_max(res)
    maxM = tf.math.reduce_min(res)
    output = tf.math.divide(tf.math.multiply(tf.math.subtract(res, minM), 255), tf.math.subtract(maxM, minM))

    return output
