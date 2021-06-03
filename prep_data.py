import os
import time
import logging
import pickle as pkl
import tensorflow as tf
import multiprocessing
from utils import setup_logger
from prep_utils import process_path, preprocess_image

log_dir = ".\\Logs\\"
logger = setup_logger("prep_logger", log_dir + "prep_data.log")
logger.error("TensorFlow version: %s", tf.__version__)

data_dir = '.\\UCF-101\\img_data'
batch_list = []  # Refers to list of batches, where 1 batch is composed of 5 images to be fed to the GAN

"""
Makes set of images from the source folder.
Example: set0 = [pic0, pic1, pic2, pic3, pic4]
         set1 = [pic1, pic2, pic3, pic4, pic5]
         ....
         set(n-4) = [pic(n-4), pic(n-3), pic(n-2), pic(n-1), pic(n)]
"""
for root, d_names, f_names in os.walk(data_dir):
    video = []
    for f in f_names:
        video.append(os.path.join(root, f))
    if len(video) > 0:
        video.sort()
        i = 0
        j = 5
        while j <= len(video):
            batch_list.append(video[i:j])
            i = i + 1
            j = j + 1

total_batches = len(batch_list)
logger.error('Total batches: %d', total_batches)


def preprocess_and_dump(itt, b_size, store_dir):
    n = itt + 1
    start = time.time()
    stt = time.time()
    for batch in batch_list[itt: itt + b_size]:
        f_img = tf.expand_dims(preprocess_image(process_path(batch[0])), 2)
        for img in batch[1:]:
            temp = tf.expand_dims(preprocess_image(process_path(img)), 2)
            f_img = tf.concat([f_img, temp], 2)

        temp_batch = tf.expand_dims(f_img, 0)
        if (batch_list.index(batch) % b_size) == 0:
            img_data = temp_batch
        else:
            img_data = tf.concat([img_data, temp_batch], 0)

        if n % 100 == 0:
            logger.error('%d batches done|Time taken per batch: %f secs', n, time.time() - stt)
            stt = time.time()
        n += 1

    logger.error("Total time taken %f secs", time.time() - start)
    start = time.time()
    num = itt // b_size
    with open(store_dir + 'batch_data' + str(num) + '.pickle', 'wb') as f:
        pkl.dump(img_data, f)
    logger.error("Data dump successful")
    logger.error('Time taken to dump data: %f secs', time.time() - start)


if __name__ == '__main__':
    os.makedirs(log_dir, exist_ok=True)
    store_dir = ".\\Batch_data\\"
    os.makedirs(store_dir, exist_ok=True)

    """
    chunk_size: used to create pickle dumps of aggregated batches
    Size of each chunk, keep 2**x based on the dataset, no.of GPUs and their capacity. 
    Example: if chunk_size = 128 and image dim are 240x320x3,then pkl dumps will be like
    (240, 320, 3, 5, 128)
    """
    chunk_size = 128
    max_chunks = total_batches//chunk_size + 1
    start = time.time()
    for i in range(0, max_chunks):
        process_eval = multiprocessing.Process(target=preprocess_and_dump, args=(i * chunk_size, chunk_size, store_dir))
        process_eval.start()
        process_eval.join()
    logging.error('Total Time taken to dump data: %f secs', time.time() - start)
