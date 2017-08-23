import os
import numpy as np
import tensorflow as tf


def save(ckpt_dir, step, saver, sess, model_name):
    if not os.path.exists(ckpt_dir):
        os.makedirs(ckpt_dir)

    saver.save(sess,
               os.path.join(ckpt_dir, model_name),
               global_step=step)


def load(ckpt_dir, saver, sess):
    print(" [*] Reading checkpoints...")

    ckpt = tf.train.get_checkpoint_state(ckpt_dir)
    if ckpt and ckpt.model_checkpoint_path:
        saver.restore(sess, ckpt.model_checkpoint_path)
        return True
    else:
        return False


def save_samples(samples, path, filename):
    if not os.path.exists(path):
        os.mkdir(path)
    np.save(os.path.join(path, filename), samples)
