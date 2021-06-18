import os

import tensorflow as tf
import pandas as pd

df = pd.read_csv('train.csv')
tf.expand_dims(df[['xmin', 'xmax', 'ymin', 'ymax']].values,axis=1)

df[['xmin', 'xmax', 'ymin', 'ymax']].values.shape
ds_bbox = tf.data.Dataset.from_tensor_slices(
    [tf.constant(x) for x in df[['xmin', 'xmax', 'ymin', 'ymax']].values])

ds_bbox = tf.data.Dataset.from_tensor_slices(
    [tf.constant(x) for x in tf.expand_dims(df[['xmin', 'xmax', 'ymin', 'ymax']].values,axis=1)])
df['class_id'].values.shape
target_ds = tf.data.Dataset.from_tensor_slices(tf.expand_dims(df['class_id'].values, axis=1).shape)
target_ds.element_spec
ds_boxsize = tf.data.Dataset.from_tensor_slices([tf.constant(1) for x in range(len(ds_bbox))])
ds_boxsize.element_spec
ds_bbox.element_spec
ds = tf.data.Dataset.zip((ds_boxsize, ds_bbox))
ds.element_spec


target_ds = tf.data.Dataset.from_tensor_slices(df['class_id'].values)

ds = tf.data.Dataset.zip((ds, ds_bbox, target_ds))







def create_tf_dataset(df, is_test=False):
    ds = tf.data.Dataset.from_tensor_slices(df["image_path"].values)

    if is_test:
        return ds
    else:
        ds_bbox = tf.data.Dataset.from_tensor_slices(
            [tf.constant(x) for x in df[['xmin', 'xmax', 'ymin', 'ymax']].values])
        target_ds = tf.data.Dataset.from_tensor_slices(df['class_id'].values)

        ds = tf.data.Dataset.zip((ds, ds_bbox, target_ds))
        return ds


def parse_image(filename, image_size, augmentation=False):
    # parts = tf.strings.split(filename, '/')
    # image_id = parts[-1]
    image = tf.io.read_file(filename)
    image = tf.image.decode_jpeg(image)
    # image = tf.image.convert_image_dtype(image, tf.float32)
    image = tf.image.resize(image, image_size,  method=tf.image.ResizeMethod.LANCZOS5, antialias=True)
    # image = tf.keras.preprocessing.image.img_to_array(image)
    return image


def prep_tf_dataset(img_path, coords, class_id, image_size, augmentation=False):
    img_tensor = parse_image(img_path, image_size=image_size)
    coords = tf.cast(coords, tf.float32)
    coords = coords/image_size[0]
    return img_tensor, coords, class_id

def ds_generator(df_path, image_size=(300,300), batch_size=32):
    '''
    input : pd.DataFrame(columns=['image_name', 'xmin', 'xmax', 'ymin', 'ymax', 'class_id'])
    output : <BatchDataset shapes: ((batch, 300, 300, 3), (batch, 4), (batch,)), types: (tf.float32, tf.float32, tf.int64)>
    '''
    df = pd.read_csv(df_path)
    ds = create_tf_dataset(df)
    ds = ds.map(lambda x, y, z: (prep_tf_dataset(x, y, z, image_size=image_size)),
                num_parallel_calls=tf.data.experimental.AUTOTUNE).prefetch(tf.data.experimental.AUTOTUNE)
    ds = ds.shuffle(3000).batch(batch_size)
    return ds


def create_batch_generator(root_dir, year, default_boxes,
                           new_size, batch_size, num_batches,
                           mode,
                           augmentation=None):
    num_examples = batch_size * num_batches if num_batches > 0 else -1
    voc = VOCDataset(root_dir, year, default_boxes,
                     new_size, num_examples, augmentation)

    info = {
        'idx_to_name': voc.idx_to_name,
        'name_to_idx': voc.name_to_idx,
        'length': len(voc),
        'image_dir': voc.image_dir,
        'anno_dir': voc.anno_dir
    }

    if mode == 'train':
        train_gen = partial(voc.generate, subset='train')
        train_dataset = tf.data.Dataset.from_generator(
            train_gen, (tf.string, tf.float32, tf.int64, tf.float32))
        val_gen = partial(voc.generate, subset='val')
        val_dataset = tf.data.Dataset.from_generator(
            val_gen, (tf.string, tf.float32, tf.int64, tf.float32))

        train_dataset = train_dataset.shuffle(40).batch(batch_size)
        val_dataset = val_dataset.batch(batch_size)

        return train_dataset.take(num_batches), val_dataset.take(-1), info
    else:
        dataset = tf.data.Dataset.from_generator(
            voc.generate, (tf.string, tf.float32, tf.int64, tf.float32))
        dataset = dataset.batch(batch_size)
        return dataset.take(num_batches), info

train_ds = ds_generator('train.csv')
val_ds = ds_generator('validation.csv')


