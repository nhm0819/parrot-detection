import tensorflow as tf
import pandas as pd


def create_tf_dataset(df, is_test=False):
    ds = tf.data.Dataset.from_tensor_slices(df['image_path'].values)

    if is_test:
        return ds
    else:
        ds_bbox = tf.data.Dataset.from_tensor_slices(
            [tf.constant(x) for x in tf.expand_dims(df[['ymin', 'xmin', 'ymax', 'xmax']].values, axis=1)])

        target_ds = tf.data.Dataset.from_tensor_slices(tf.expand_dims(df['class_id'].values, axis=1))

        ds = tf.data.Dataset.zip((ds, ds_bbox, target_ds))
        return ds


def parse_image(filename, image_size):
    # parts = tf.strings.split(filename, '/')
    # image_id = parts[-1]
    image = tf.io.read_file(filename)
    image = tf.image.decode_jpeg(image)
    # image = tf.image.convert_image_dtype(image, tf.float32)
    image = tf.image.resize(image, image_size,  method=tf.image.ResizeMethod.LANCZOS5, antialias=True)
    # image = tf.keras.preprocessing.image.img_to_array(image)
    return image


def prep_tf_dataset(img_path, gt_boxes, class_id, image_size, augmentation):
    img_tensor = parse_image(img_path, image_size=image_size)
    gt_boxes = tf.cast(gt_boxes, tf.float32)
    gt_boxes = gt_boxes/image_size[0]

    # class_id = tf.cast(class_id + 1, tf.int32)

    if augmentation:
        img, gt_boxes = augmentation(img_tensor, gt_boxes)

    return img_tensor, gt_boxes, class_id

def ds_generator(df_path, image_size=(300,300), augmentation=False):
    '''
    input : pd.DataFrame(columns=['image_name', 'ymin', 'xmin', 'ymax', 'xmax', 'class_id'])
    output : <BatchDataset shapes: ((batch, 300, 300, 3), (batch, 4), (batch,)), types: (tf.float32, tf.float32, tf.int64)>
    '''
    df = pd.read_csv(df_path)
    data_len = len(df)
    ds = create_tf_dataset(df)
    ds = ds.map(lambda x, y, z: (prep_tf_dataset(x, y, z, image_size=image_size, augmentation=augmentation)),
                num_parallel_calls=tf.data.experimental.AUTOTUNE).prefetch(tf.data.experimental.AUTOTUNE)
    return ds, data_len


def get_data_shapes():
    """Generating data shapes for tensorflow datasets.
    outputs:
        data shapes = output data shapes for (images, ground truth boxes, ground truth labels)
    """
    return ([None, None, None], [None, None], [None])


def get_padding_values():
    """Generating padding values for missing values in batch for tensorflow datasets.
    outputs:
        padding values = padding values with dtypes for (images, ground truth boxes, ground truth labels)
    """
    return (tf.constant(0, tf.float32), tf.constant(0, tf.float32), tf.constant(-1, tf.int64))