import tensorflow as tf
import numpy as np


def bbox_overlap_iou(bboxes1, bboxes2):
    x11, y11, x12, y12 = tf.split(bboxes1, 4, axis=2)
    x21, y21, x22, y22 = tf.split(bboxes2, 4, axis=2)

    x11 = tf.transpose(x11, (1, 2, 0))
    y11 = tf.transpose(y11, (1, 2, 0))
    x12 = tf.transpose(x12, (1, 2, 0))
    y12 = tf.transpose(y12, (1, 2, 0))

    x21 = tf.transpose(x21, (1, 2, 0))
    y21 = tf.transpose(y21, (1, 2, 0))
    x22 = tf.transpose(x22, (1, 2, 0))
    y22 = tf.transpose(y22, (1, 2, 0))

    xI1 = tf.maximum(x11, tf.transpose(x21))
    yI1 = tf.maximum(y11, tf.transpose(y21))

    xI2 = tf.minimum(x12, tf.transpose(x22))
    yI2 = tf.minimum(y12, tf.transpose(y22))

    inter_area = (xI2 - xI1 + 1) * (yI2 - yI1 + 1)

    bboxes1_area = (x12 - x11 + 1) * (y12 - y11 + 1)
    bboxes2_area = (x22 - x21 + 1) * (y22 - y21 + 1)

    union = (bboxes1_area + tf.transpose(bboxes2_area)) - inter_area

    return tf.transpose(tf.maximum(inter_area / union, 0.0), (1, 0, 2))


def bbox_iog(ground_truth, predicted):
    x11, y11, x12, y12 = tf.split(predicted, 4, axis=2)
    x21, y21, x22, y22 = tf.split(ground_truth, 4, axis=2)

    xI1 = tf.maximum(x11, tf.transpose(x21))
    yI1 = tf.maximum(y11, tf.transpose(y21))

    xI2 = tf.minimum(x12, tf.transpose(x22))
    yI2 = tf.minimum(y12, tf.transpose(y22))

    intersect_area = (xI2 - xI1 + 1) * (yI2 - yI1 + 1)

    gt_area = (x22 - x21 + 1) * (y22 - y21 + 1)

    return tf.transpose(tf.maximum(intersect_area / gt_area, 0.0), (1, 0, 2))


def smooth_l1_distance(y_true, y_pred, smooth):
    sigma_squared = smooth ** 2
    regression_diff = y_pred - y_true
    regression_diff = tf.abs(regression_diff)
    return tf.where(
        tf.less(regression_diff, 1.0 / sigma_squared),
        0.5 * sigma_squared * tf.pow(regression_diff, 2),
        regression_diff - 0.5 / sigma_squared
    )


def smooth_ln(x, smooth):
    return tf.where(
        tf.less_equal(x, smooth),
        -tf.log(1 - x),
        ((x - smooth) / (1 - smooth)) - tf.log(1 - smooth)
    )


def attraction_term(y_pred, ious_over_truth_boxes, smooth=0.5):
    iou_max_indices = tf.argmax(ious_over_truth_boxes[..., 0], axis=2)
    ious_over_truth_boxes = ious_over_truth_boxes[..., 1:]

    gt_boxes_with_max_ious = None
    for batch_num in np.arange(y_pred.shape[0].value, dtype=np.int64):
        indices = tf.stack([tf.cast(tf.tile([batch_num], [tf.shape(y_pred)[1]]), dtype=tf.int64),
                            tf.range(tf.cast(tf.shape(y_pred)[1], dtype=tf.int64)),
                            tf.cast(iou_max_indices[batch_num], dtype=tf.int64)])
        indices = tf.transpose(indices)

        if gt_boxes_with_max_ious is None:
            gt_boxes_with_max_ious = tf.expand_dims(tf.gather_nd(ious_over_truth_boxes, indices), axis=0)
        else:
            boxes_with_max_ious = tf.expand_dims(tf.gather_nd(ious_over_truth_boxes, indices), axis=0)
            gt_boxes_with_max_ious = tf.concat([gt_boxes_with_max_ious, boxes_with_max_ious], axis=0)

    l1_distances = smooth_l1_distance(y_pred, gt_boxes_with_max_ious, smooth)
    return tf.reduce_sum(tf.cast(l1_distances, tf.float32)) / tf.cast(tf.shape(y_pred)[1], dtype=tf.float32)


def repulsion_term_gt(y_pred, ious_over_truth_boxes, smooth=0.99):
    _, indices_2highest_iou = tf.nn.top_k(ious_over_truth_boxes[..., 0], k=2)
    ious_over_truth_boxes = ious_over_truth_boxes[..., 1:]
    indices_2highest_iou = indices_2highest_iou[..., 1]

    gt_boxes_with_max_ious = None
    for batch_num in np.arange(y_pred.shape[0].value, dtype=np.int64):
        indices = tf.stack([tf.cast(tf.tile([batch_num], [tf.shape(y_pred)[1]]), dtype=tf.int64),
                            tf.range(tf.cast(tf.shape(y_pred)[1], dtype=tf.int64)),
                            tf.cast(indices_2highest_iou[batch_num], dtype=tf.int64)])
        indices = tf.transpose(indices)

        if gt_boxes_with_max_ious is None:
            gt_boxes_with_max_ious = tf.expand_dims(tf.gather_nd(ious_over_truth_boxes, indices), axis=0)
        else:
            boxes_with_max_ious = tf.expand_dims(tf.gather_nd(ious_over_truth_boxes, indices), axis=0)
            gt_boxes_with_max_ious = tf.concat([gt_boxes_with_max_ious, boxes_with_max_ious], axis=0)

    ln_distances_for_iog = smooth_ln(bbox_iog(gt_boxes_with_max_ious, y_pred), smooth)
    return tf.reduce_sum(ln_distances_for_iog) / tf.cast(tf.shape(y_pred)[1], dtype=tf.float32)


def repulsion_term_box(ious, smooth=0.01):
    iou_over_predicted_indices = tf.where(tf.less(ious, 1.0))
    ious = tf.gather_nd(ious, iou_over_predicted_indices)

    dist_sum = tf.reduce_sum(smooth_ln(ious, smooth))
    iou_sum = tf.reduce_sum(ious)

    return dist_sum / tf.maximum(iou_sum, 0.000001)


def create_repulsion_loss(alpha=0.5, betta=0.5):
    def _filter_predictions(y_pred):
        y_pred_indices = tf.where(tf.greater_equal(y_pred[..., 4], 0.5))
        return tf.gather_nd(y_pred, [y_pred_indices])

    def _preprocess_inputs(y_true, y_pred):
        return y_true[..., :4], y_pred[..., :4]

    def _repulsion_impl(y_true, y_pred):
        len_of_predicted = tf.shape(y_pred)[1]

        ious = bbox_overlap_iou(y_pred, y_true)
        tiled_for_concat = tf.tile(tf.expand_dims(y_true, axis=1), [1, len_of_predicted, 1, 1])
        ious_over_truth_boxes = tf.concat([tf.expand_dims(ious, axis=3), tiled_for_concat], axis=3)

        return tf.reduce_sum([
            attraction_term(y_pred, ious_over_truth_boxes),
            alpha * repulsion_term_gt(y_pred, ious_over_truth_boxes),
            betta * repulsion_term_box(ious)
        ])

    def _repulsion_loss(y_true, y_pred):
        # Фильтруем y_pred, оставляя те, у которых IOU > 0,5 хотябы с одним y_true

        y_pred = _filter_predictions(y_pred)
        y_true, y_pred = _preprocess_inputs(y_true, y_pred)

        return tf.cond(tf.logical_or(tf.equal(tf.shape(y_pred)[1], 0), tf.equal(tf.shape(y_true)[1], 0)),
                       lambda: tf.Variable(0.0, dtype=tf.float32),
                       lambda: _repulsion_impl(y_true, y_pred))

    return _repulsion_loss
