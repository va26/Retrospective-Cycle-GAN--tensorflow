import tensorflow as tf
import model
import os
import datetime
import time
import argparse
import numpy as np
import utils

log_dir = r".\\Logs\\"
os.makedirs(log_dir, exist_ok=True)

# brief logger
logger = utils.setup_logger('first logger', log_dir + 'train brief.log')

# detailed logger
super_logger = utils.setup_logger('second logger', log_dir + 'train_detail.log')
super_logger.error("TensorFlow version: %s", tf.__version__)

AUTOTUNE = tf.data.experimental.AUTOTUNE
parser = argparse.ArgumentParser(description="Enter training hyper-parameters...")
parser.add_argument('--batch', type=int, default=1, help='BATCH_SIZE for training <= 2')
parser.add_argument('--max-ckpt', type=int, default=5, help='Max to keep checkpoints while training')
parser.add_argument('--epoch', type=int, default=20, help='Epoch count for training')
parser.add_argument('--gpu-ids', type=int, nargs="+", default=0, help='List of ids of GPUs to use')
parser.add_argument('--tb-log-dir', type=str, default=".\\Logs\\tensorboard", help='Directory for tensorboard logging')
parser.add_argument('--ckpt-dir', type=str, default=".\\checkpoints\\",
                    help='Directory to (re-)store checkpoints for training')
parser.add_argument('--filter-size', type=int, default=7, help='Filter size for Laplacian of Gaussian (LoG)')
parser.add_argument('--sigma', type=int, default=1, help='Sigma for Laplacian of Gaussian (LoG)')
args = parser.parse_args()


BATCH_SIZE_PER_REPLICA = args.b
dir_ = ".\\Batch_data\\"
max_chunks = len([name for name in os.listdir(dir_) if os.path.isfile(name)])

super_logger.error('BATCH_SIZE_PER_REPLICA: %d', BATCH_SIZE_PER_REPLICA)

# If the list of devices is not specified in the
# 'tf.distribute.MirroredStrategy' constructor, it will be auto-detected

if isinstance(args.gpu_ids, int):
    args.gpu_ids = [args.gpu_ids]
if len(args.gpu_ids) > 1:
    device_list = ['/gpu:' + str(x) for x in args.gpu_ids]
elif len(args.gpu_ids) == 1:
    device_list = None

strategy = tf.distribute.MirroredStrategy(devices=device_list, cross_device_ops=tf.distribute.HierarchicalCopyAllReduce())
super_logger.error('Number of devices: %d', strategy.num_replicas_in_sync)

GLOBAL_BATCH_SIZE = BATCH_SIZE_PER_REPLICA * strategy.num_replicas_in_sync
LAMBDA1 = 0.005
LAMBDA2 = 0.003
LAMBDA3 = 0.003
EPOCHS = args.epoch


with strategy.scope():
    OUTPUT_CHANNELS = 3

    generator = model.generator(OUTPUT_CHANNELS, norm_type='instancenorm')

    discriminator_frame = model.discriminator(norm_type='instancenorm', target=False)

    discriminator_seq = model.discriminator_seq(norm_type='instancenorm', target=False)


with strategy.scope():
    loss_obj_11 = tf.keras.losses.MeanAbsoluteError(reduction=tf.keras.losses.Reduction.NONE)
    loss_obj_12 = tf.keras.losses.MeanSquaredError(reduction=tf.keras.losses.Reduction.NONE)

with strategy.scope():
    def discriminator_loss(real, generated):
        real = tf.squeeze(real, 3)
        generated = tf.squeeze(generated, 3)
        real_loss = loss_obj_12(tf.ones_like(real), real)

        generated_loss = loss_obj_12(tf.zeros_like(generated), generated)

        total_disc_loss = real_loss + generated_loss

        return tf.nn.compute_average_loss(total_disc_loss, global_batch_size=GLOBAL_BATCH_SIZE)

with strategy.scope():
    def generator_loss_reconstruction(image1, image2):
        if len(image1.shape) > 3:
            image1 = tf.squeeze(image1, 3)

        if len(image2.shape) > 3:
            image2 = tf.squeeze(image2, 3)
        gen_loss_rec = loss_obj_11(image1, image2)

        return tf.nn.compute_average_loss(gen_loss_rec, global_batch_size=GLOBAL_BATCH_SIZE)

"""
with strategy.scope():
    def generator_loss_adversarial(generated):
        generated = tf.squeeze(generated, 3)
        gen_loss_adv = loss_obj_12(tf.ones_like(generated), generated)
        return tf.nn.compute_average_loss(gen_loss_adv, global_batch_size=GLOBAL_BATCH_SIZE)
"""

with strategy.scope():
    generator_optimizer = tf.keras.optimizers.Adam(3e-4, beta_1=0.5, beta_2=0.999)
    discriminator_frame_optimizer = tf.keras.optimizers.Adam(3e-4, beta_1=0.5, beta_2=0.999)
    discriminator_seq_optimizer = tf.keras.optimizers.Adam(3e-4, beta_1=0.5, beta_2=0.999)

with strategy.scope():
    checkpoint_path = args.ckpt_dir

    ckpt = tf.train.Checkpoint(generator=generator,
                               discriminator_frame=discriminator_frame,
                               discriminator_seq=discriminator_seq,
                               generator_optimizer=generator_optimizer,
                               discriminator_frame_optimizer=discriminator_frame_optimizer,
                               discriminator_seq_optimizer=discriminator_seq_optimizer)

    ckpt_manager = tf.train.CheckpointManager(ckpt, checkpoint_path, max_to_keep=args.max_ckpt)

    # if a checkpoint exists, restore the latest checkpoint.
    if ckpt_manager.latest_checkpoint:
        ckpt.restore(ckpt_manager.latest_checkpoint)
        super_logger.error('Latest Checkpoint restored!! - %s', ckpt_manager.latest_checkpoint.split('\\')[-1])
        logger.error('Latest Checkpoint restored!! - %s', ckpt_manager.latest_checkpoint.split('\\')[-1])


def interim_step(batch):
    xn = tf.expand_dims(batch[:, :, :, 4, :], 3)
    forward_series = batch[:, :, :, :4, :]
    disc_xn_frame = discriminator_frame(xn, training=True)
    xn_log = utils.laplacian_of_gaussian(xn, args.filter_size, sigma=args.sigma)
    xn_log = tf.convert_to_tensor(xn_log)
    xn1, xn1_log, disc_xn1_frame = sub_interim_step(forward_series)

    return xn, xn_log, xn1, xn1_log, disc_xn_frame, disc_xn1_frame


def sub_interim_step(forward_series):
    xn1 = generator(forward_series, training=True)
    xn1_log = utils.laplacian_of_gaussian(xn1, args.filter_size, sigma=args.sigma)
    xn1_log = tf.convert_to_tensor(xn1_log)
    disc_xn1_frame = discriminator_frame(xn1, training=True)

    return xn1, xn1_log, disc_xn1_frame


def train_step(batch):
    # persistent is set to True because the tape is used more than
    # once to calculate the gradients.

    with tf.GradientTape(persistent=True) as tape:
        # Generator G translates X -> Y
        # Generator F translates Y -> X.

        # Forward
        xn, xn_log, xn1, xn1_log, disc_xn_frame, disc_xn1_frame = interim_step(batch)

        # Backward
        xm, xm_log, xm1, xm1_log, disc_xm_frame, disc_xm1_frame = interim_step(batch[:, :, :, ::-1, :])

        fake_forward = tf.concat([xm1, batch[:, :, :, 1:4, :]], 3)  # func([xm1[0], x2, x3, x4], axis)
        xn2, xn2_log, disc_xn2_frame = sub_interim_step(fake_forward)

        fake_backward = tf.concat([xn1, batch[:, :, :, ::-1, :][:, :, :, 1:4, :]], 3)
        xm2, xm2_log, disc_xm2_frame = sub_interim_step(fake_backward)

        full_forward = batch  # func([x1, x2, x3, x4, x5], axis)
        full_forward_fake = tf.concat([batch[:, :, :, :4, :], xn1], 3)  # func([x1, x2, x3, x4, xn1[0]], axis)
        full_forward_fake_2 = tf.concat([batch[:, :, :, :4, :], xn2], 3)  # func([x1, x2, x3, x4, xn2[0]], axis)

        full_backward = batch[:, :, :, ::-1, :]  # func([x5, x4, x3, x2, x1], axis)
        full_backward_fake = tf.concat([full_backward[:, :, :, :4, :], xm1], 3)  # func([x5, x4, x3, x2, xm1[0]], axis)
        full_backward_fake_2 = tf.concat([full_backward[:, :, :, :4, :], xm2],
                                         3)  # func([x5, x4, x3, x2, xm2[0]], axis)

        disc_xn_seq = discriminator_seq(full_forward, training=True)
        disc_xn1_seq = discriminator_seq(full_forward_fake, training=True)
        disc_xn2_seq = discriminator_seq(full_forward_fake_2, training=True)

        disc_xm_seq = discriminator_seq(full_backward, training=True)
        disc_xm1_seq = discriminator_seq(full_backward_fake, training=True)
        disc_xm2_seq = discriminator_seq(full_backward_fake_2, training=True)

        # (xm, xm1, xm2), (xn, xn1, xn2)
        # (xm_log, xm1_log, xm2_log), (xn_log, xn1_log, xn2_log)
        loss_list = [(xm, xm1), (xm1, xm2), (xm, xm2), (xn, xn1), (xn1, xn2), (xn, xn2)]
        log_loss_list = [(xm_log, xm1_log), (xm1_log, xm2_log), (xm_log, xm2_log), (xn_log, xn1_log),
                         (xn1_log, xn2_log), (xn_log, xn2_log)]

        disc_frame_loss_list = [(disc_xn_frame, disc_xn1_frame), (disc_xn_frame, disc_xn2_frame),
                                (disc_xn1_frame, disc_xn2_frame),
                                (disc_xm_frame, disc_xm1_frame), (disc_xm_frame, disc_xm2_frame),
                                (disc_xm1_frame, disc_xm2_frame)]

        disc_seq_loss_list = [(disc_xn_seq, disc_xn1_seq), (disc_xn_seq, disc_xn2_seq), (disc_xn1_seq, disc_xn2_seq),
                              (disc_xm_seq, disc_xm1_seq), (disc_xm_seq, disc_xm2_seq), (disc_xm1_seq, disc_xm2_seq)]

        gen_loss = sum([generator_loss_reconstruction(x[0], x[1]) for x in loss_list])
        gen_log_loss = sum([generator_loss_reconstruction(x[0], x[1]) for x in log_loss_list])

        adversarial_frame_loss = sum([discriminator_loss(x[0], x[1]) for x in disc_frame_loss_list])
        adversarial_seq_loss = sum([discriminator_loss(x[0], x[1]) for x in disc_seq_loss_list])

        # Total Loss = Adversarial Loss + Generator Loss
        total_gen_loss = gen_loss + (LAMBDA1 * gen_log_loss) + (LAMBDA2 * adversarial_frame_loss) + (
                LAMBDA3 * adversarial_seq_loss)
        disc_frame_loss = adversarial_frame_loss
        disc_seq_loss = adversarial_seq_loss

    # Calculate gradients for generator and discriminator
    gen_gradients = tape.gradient(total_gen_loss, generator.trainable_variables)
    disc_frame_gradients = tape.gradient(disc_frame_loss, discriminator_frame.trainable_variables)
    disc_seq_gradients = tape.gradient(disc_seq_loss, discriminator_seq.trainable_variables)

    # Apply gradients to optimizers
    generator_optimizer.apply_gradients(zip(gen_gradients, generator.trainable_variables))
    discriminator_frame_optimizer.apply_gradients(zip(disc_frame_gradients, discriminator_frame.trainable_variables))
    discriminator_seq_optimizer.apply_gradients(zip(disc_seq_gradients, discriminator_seq.trainable_variables))

    return total_gen_loss, disc_frame_loss, disc_seq_loss


def tensorboard_log(log_dir):
    tb_callback = tf.keras.callbacks.TensorBoard(log_dir + '\\models')
    tb_callback.set_model(generator)
    tb_callback.set_model(discriminator_frame)
    tb_callback.set_model(discriminator_seq)
    summary_writer = tf.summary.create_file_writer(log_dir + '\\train\\' +
                                                   datetime.datetime.now().strftime("%Ym&d-%H%M%S"))

    return summary_writer


@tf.function
def distributed_train_step(dataset_inputs):
    gen_loss, disc_frame_loss, disc_seq_loss = strategy.run(train_step, args=(dataset_inputs,))
    gen_loss = strategy.reduce(tf.distribute.ReduceOp.SUM, gen_loss, axis=None)
    disc_frame_loss = strategy.reduce(tf.distribute.ReduceOp.SUM, disc_frame_loss, axis=None)
    disc_seq_loss = strategy.reduce(tf.distribute.ReduceOp.SUM, disc_seq_loss, axis=None)

    return gen_loss, disc_frame_loss, disc_seq_loss


tb_log_dir = args.tb_log_dir  # In case if training is done on a server then a different path to be specified
tb_local_dir = args.tb_log_dir
writer_local = tensorboard_log(tb_local_dir)
writer = tensorboard_log(tb_log_dir)

logger.error("Training Begins........")

# @tf.function
for epoch in range(EPOCHS):
    start = time.time()

    n = 1
    loss_arr = []
    for i in range(max_chunks):  # Full dataset comprises of #63
        train_dist_dataset = utils.load_data(dir_ + "batch_data" + str(i) + ".pickle", GLOBAL_BATCH_SIZE, strategy)
        for batch in train_dist_dataset:
            st = time.time()
            gen_loss, disc_frame_loss, disc_seq_loss = distributed_train_step(batch)

            loss = [gen_loss, disc_frame_loss, disc_seq_loss]
            loss_arr.append(loss)
            n += 1

        super_logger.error('%s done in %f secs', ("batch data" + str(i)), (time.time() - st))

    loss_arr = np.asarray(loss_arr)
    total_gen_loss, total_disc_frame_loss, total_disc_seq_loss = loss_arr.sum(axis=0)

    super_logger.error('total_gen_loss for epoch %d is %f', (epoch + 1), total_gen_loss)
    logger.error('total_gen_loss for epoch %d is %f', (epoch + 1), total_gen_loss)
    super_logger.error('total_disc_frame_loss for epoch %d is %f', (epoch + 1), total_disc_frame_loss)
    logger.error('total_disc_frame_loss for epoch %d is %f', (epoch + 1), total_disc_frame_loss)
    super_logger.error('total_disc_seq_loss for epoch %d is %f', (epoch + 1), total_disc_seq_loss)
    logger.error('total_disc_seq_loss for epoch %d is %f', (epoch + 1), total_disc_seq_loss)

    with writer_local.as_default():
        tf.summary.scalar('total_gen_loss', total_gen_loss, step=(epoch + 1))
        tf.summary.scalar('total_disc_frame_loss', total_disc_frame_loss, step=(epoch + 1))
        tf.summary.scalar('total_disc_seq_loss', total_disc_seq_loss, step=(epoch + 1))
    writer_local.flush()

    with writer.as_default():
        tf.summary.scalar('total_gen_loss', total_gen_loss, step=(epoch + 1))
        tf.summary.scalar('total_disc_frame_loss', total_disc_frame_loss, step=(epoch + 1))
        tf.summary.scalar('total_disc_seq_loss', total_disc_seq_loss, step=(epoch + 1))
    writer.flush()

    if (epoch + 1) % 5 == 0:
        ckpt_save_path = ckpt_manager.save()
        super_logger.error('Saving checkpoint for epoch %d at %s', (epoch + 1), ckpt_save_path)
        logger.error('Saving checkpoint for epoch %d at %s', (epoch + 1), ckpt_save_path)

    super_logger.error('Total Batches for an epoch per GPU: %d', n)
    super_logger.error('Time taken for an epoch %d is %.4f secs', (epoch + 1), time.time() - start)
    logger.error('Time taken for an epoch %d is %.2f hrs', (epoch + 1), (time.time() - start) / 3600)
