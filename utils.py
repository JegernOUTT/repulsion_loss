import cv2
import matplotlib.pyplot as plt
import numpy as np


def _fix_boxes(boxes):
    return np.array([
        [max(box[0], 0.0), max(box[1], 0.0), min(box[2], 1.0), min(box[3], 1.0)]
        for box in boxes
    ], dtype=np.float)


def get_boxes(similarity=0.5, seed=42):
    y_true = np.array([[0.15520833333333334, 0.43796296296296294, 0.2947916666666667, 0.9202938475665748],
                       [0.5661458333333333, 0.2537037037037037, 0.6270833333333333, 0.5999488438714958],
                       [0.5901041666666667, 0.2324074074074074, 0.6453125, 0.5045784735011255],
                       [0.7692708333333333, 0.2898148148148148, 0.8046875, 0.3620072644992259],
                       [0.05, 0.6898148148148148, 0.17395833333333333, 0.9721456994184267],
                       [0.0140625, 0.4434419957147229, 0.11302083333333333, 0.9333333333333333],
                       [0.8442708333333333, 0.30366528354080224, 0.8791666666666667, 0.38425925925925924],
                       [0.165625, 0.27311632938780706, 0.18854166666666666, 0.32314814814814813],
                       [0.8411458333333334, 0.8866598079561042, 0.9417181069958848, 1.0],
                       [0.5390625, 0.2758747697974217, 0.5786458333333333, 0.5509259259259259],
                       [0.115625, 0.29719040346188114, 0.13697916666666668, 0.41200521827669595],
                       [0.8421875, 0.2990356539111726, 0.8723958333333334, 0.37125787613339484],
                       [0.7765625, 0.2508961533881148, 0.8052083333333333, 0.3258961533881148],
                       [0.6104166666666667, 0.1842294867214481, 0.621875, 0.24256282005478144],
                       [0.6203125, 0.17774800523996662, 0.6286458333333333, 0.2138591163510777],
                       [0.32083333333333336, 0.20645071982281288, 0.3333333333333333, 0.2620062753783684]],
                      dtype=np.float)

    y_pred = np.repeat(y_true, repeats=100, axis=0)
    np.random.seed(seed)
    random_shifts = np.random.rand(*y_pred.shape) + similarity
    random_shifts[random_shifts > 1.] = 1.
    y_pred = y_pred * random_shifts
    return y_true, _fix_boxes(y_pred)


def draw_boxes(true_boxes, predicted_boxes):
    image = np.full((1000, 1000, 3), 0, dtype=np.int8)

    true_boxes_color = 0, 255, 0
    predicted_boxes_color = 255, 0, 0

    true_boxes = true_boxes.copy()
    true_boxes *= 1000
    true_boxes = true_boxes.astype(np.int)

    predicted_boxes = predicted_boxes.copy()
    predicted_boxes *= 1000
    predicted_boxes = predicted_boxes.astype(np.int)

    for box in true_boxes:
        cv2.rectangle(image, (box[0], box[1]), (box[2], box[3]), true_boxes_color, 4)

    for box in predicted_boxes:
        cv2.rectangle(image, (box[0], box[1]), (box[2], box[3]), predicted_boxes_color, 1)

    plt.figure(figsize=(20, 20))
    return plt.imshow(image)

