import tensorflow as tf

IMG_WIDTH = 240
IMG_HEIGHT = 320


def process_path(file_path):
    # label = get_label(file_path)
    # load the raw data from the file as a string
    img = tf.io.read_file(file_path)
    img = decode_img(img)
    return img


def decode_img(img):
    # convert the compressed string to a 3D uint8 tensor
    img = tf.image.decode_jpeg(img, channels=3)
    return tf.image.resize(img, [IMG_WIDTH, IMG_HEIGHT])


def normalize(image):
    image = tf.cast(image, tf.float32)
    image = (image / 127.5) - 1
    return image


def preprocess_image(image):
    # image = random_jitter(image)
    image = normalize(image)
    return image
