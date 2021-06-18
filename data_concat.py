import os
import pandas as pd
import shutil

dataset_dir = os.path.join(os.getcwd(), "dataset")

dataset_classes = os.listdir(dataset_dir)
dataset_annotations_dir = [os.path.join(dataset_dir, cls, "csv") for cls in dataset_classes]


# search csv files
csv_filenames = []
def search(dirname):
    try:
        global csv_filenames
        filenames = os.listdir(dirname)
        for filename in filenames:
            full_filename = os.path.join(dirname, filename)
            if os.path.isdir(full_filename):
                search(full_filename)
            else:
                ext = os.path.splitext(full_filename)[-1]
                if ext == '.csv':
                    # print(full_filename)
                    csv_filenames.append(full_filename)
    except PermissionError:
        pass

search(dataset_dir)


# Ground truth
train_labels_filenames = [file for file in csv_filenames if file.endswith("train.csv")]
val_labels_filenames = [file for file in csv_filenames if file.endswith("val.csv")]
test_labels_filenames = [file for file in csv_filenames if file.endswith("test.csv")]

#%%
train_df = pd.DataFrame(columns=['frame', 'xmin', 'xmax', 'ymin', 'ymax', 'class_id'])
for train_labels_filename in train_labels_filenames:
    train_df = pd.concat([train_df, pd.read_csv(train_labels_filename)])
train_df = train_df.rename(columns={'frame' : 'image_path'})
train_df['image_path'] = train_df['image_path'].map(
    lambda x : os.path.join([cls for cls in dataset_classes if cls.startswith(x.split('_')[0])][0], x))

val_df = pd.DataFrame(columns=['frame', 'xmin', 'xmax', 'ymin', 'ymax', 'class_id'])
for val_labels_filename in val_labels_filenames:
    val_df = pd.concat([val_df, pd.read_csv(val_labels_filename)])
val_df = val_df.rename(columns={'frame': 'image_path'})
val_df['image_path'] = val_df['image_path'].map(
    lambda x : os.path.join([cls for cls in dataset_classes if cls.startswith(x.split('_')[0])][0], x))

test_df = pd.DataFrame(columns=['frame', 'xmin', 'xmax', 'ymin', 'ymax', 'class_id'])
for test_labels_filename in test_labels_filenames:
    test_df = pd.concat([test_df, pd.read_csv(test_labels_filename)])
test_df = test_df.rename(columns={'frame' : 'image_path'})
test_df['image_path'] = test_df['image_path'].map(
    lambda x : os.path.join([cls for cls in dataset_classes if cls.startswith(x.split('_')[0])][0], x))

#%%
train_df['image_path'] = 'dataset\\' + train_df['image_path']
val_df['image_path'] = 'dataset\\' + val_df['image_path']
test_df['image_path'] = 'dataset\\' + test_df['image_path']


#%%
train_df.to_csv('train.csv', encoding='utf-8-sig', index=None)
val_df.to_csv('validation.csv', encoding='utf-8-sig', index=None)
test_df.to_csv('test.csv', encoding='utf-8-sig', index=None)


for cls in dataset_classes:
    images = os.listdir(os.path.join(dataset_dir, cls, 'img'))
    for img in images:
        path = os.path.join(dataset_dir, cls, 'img', img)
        shutil.move(path, os.path.join(dataset_dir, cls))
    shutil.rmtree(os.path.join(dataset_dir, cls, 'img'))