import numpy as np
import tensorflow as tf


def crop_and_resize(featuremap, boxes, box_ind, crop_size, pad_border=True):
    """
    Aligned version of tf.image.crop_and_resize, following our definition of floating point boxes.

    Args:
        featuremap: NHWC
        boxes: nx4, x1y1x2y2
        box_ind: (n,)
        crop_size (int):
    Returns:
        n,C,size,size
    """
    assert isinstance(crop_size, int), crop_size
    boxes = tf.stop_gradient(boxes)

    # TF's crop_and_resize produces zeros on border
    if pad_border:
        # this can be quite slow
        featuremap = tf.pad(featuremap, [[0, 0], [1, 1], [1, 1], [0, 0]], mode='SYMMETRIC')
        boxes = boxes + 1

    def transform_fpcoor_for_tf(boxes, featuremap_shape, crop_shape):
        x0, y0, x1, y1 = tf.split(boxes, 4, axis=1)

        spacing_w = (x1 - x0) / tf.to_float(crop_shape[1])
        spacing_h = (y1 - y0) / tf.to_float(crop_shape[0])

        nx0 = (x0 + spacing_w / 2 - 0.5) / tf.to_float(featuremap_shape[1] - 1)
        ny0 = (y0 + spacing_h / 2 - 0.5) / tf.to_float(featuremap_shape[0] - 1)

        nw = spacing_w * tf.to_float(crop_shape[1] - 1) / tf.to_float(featuremap_shape[1] - 1)
        nh = spacing_h * tf.to_float(crop_shape[0] - 1) / tf.to_float(featuremap_shape[0] - 1)

        return tf.concat([ny0, nx0, ny0 + nh, nx0 + nw], axis=1)

    featuremap_shape = tf.shape(featuremap)[1:3]
    boxes = transform_fpcoor_for_tf(boxes, featuremap_shape, [crop_size, crop_size])
    # image = tf.transpose(image, [0, 2, 3, 1])   # nhwc
    ret = tf.image.crop_and_resize(
        featuremap, boxes, tf.to_int32(box_ind),
        crop_size=[crop_size, crop_size])
    # ret = tf.transpose(ret, [0, 3, 1, 2])   # ncss
    return ret


def roi_align(featuremap, rois, _stride, resolution):
    """
    Args:
        featuremap: 1xHxWxC
        boxes: Nx5 floatbox
        resolution: output spatial resolution

    Returns:
        NxCx res x res
    """
    batch_ids = tf.squeeze(tf.slice(rois, [0, 0], [-1, 1], name="batch_id"), [1])
    _, x1, y1, x2, y2 = tf.unstack(rois, axis=1)
    boxes = tf.stack([x1, y1, x2, y2], axis=1)
    boxes_on_featuremap = boxes * (1.0 / _stride)
    # sample 4 locations per roi bin
    ret = crop_and_resize(
        featuremap, boxes_on_featuremap,
        batch_ids,
        resolution * 2)
    ret = tf.nn.avg_pool(ret, [1, 2, 2, 1], [1, 2, 2, 1], padding='SAME', data_format='NHWC')
    return ret


if __name__ == '__main__':
    import tensorflow.contrib.eager as tfe
    tfe.enable_eager_execution()

    # want to crop 2x2 out of a 5x5 image, and resize to 4x4
    image = np.arange(25).astype('float32').reshape(5, 5)
    boxes = np.asarray([[0, 1, 1, 3, 3]], dtype='float32')
    target = 4

    print(roi_align(image[None, :, :None], boxes, target)[0][0])
    # print(crop_and_resize(
    #     image[None, None, :, :], boxes, [0], target)[0][0])
    """
    Expected values:
    4.5 5 5.5 6
    7 7.5 8 8.5
    9.5 10 10.5 11
    12 12.5 13 13.5
    """