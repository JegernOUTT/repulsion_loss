import os

import tensorflow as tf
import keras.backend as K

from repulsion_loss import create_repulsion_loss
from utils import *

os.environ['CUDA_VISIBLE_DEVICES'] = ""


def main():
    y_true, y_pred = get_boxes(similarity=0.8)
    draw_boxes(y_true, y_pred)

    # Create extra dim for batch emulation
    y_true = np.expand_dims(y_true, axis=0)[..., :4]
    y_pred = np.expand_dims(y_pred, axis=0)[..., :4]

    # Create fake objectness score for predictions
    y_pred = np.concatenate([y_pred, np.random.sample((y_pred.shape[0], y_pred.shape[1], 1))], axis=2)

    y_true_tensor = tf.convert_to_tensor(y_true, dtype=tf.float32)
    y_pred_tensor = tf.convert_to_tensor(y_pred, dtype=tf.float32)

    repulsion_loss = create_repulsion_loss()
    gradients = tf.gradients(repulsion_loss(y_true_tensor, y_pred_tensor),
                             [y_true_tensor, y_pred_tensor])

    print('Repulsion_loss: ', K.eval(repulsion_loss(y_true_tensor, y_pred_tensor)))
    print('Gradients: ', list(map(K.eval, gradients)))


if __name__ == '__main__':
    main()
