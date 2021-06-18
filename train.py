import os
from datetime import datetime
from tensorflow.keras.optimizers import SGD, Adam
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard, LearningRateScheduler
from ssd_loss import CustomLoss
from models.ssd_vgg16 import get_model, init_model
from utils import preprocessing, augmentation
from utils.bbox_utils import generate_prior_boxes
from utils.train_utils import generator, scheduler, get_step_size



# dataset directory
dataset_dir = os.path.join(os.getcwd(), "dataset")


# hyper parameters config
hyper_params = {}
hyper_params["image_size"] = (300, 300)
hyper_params["aspect_ratios"] = [[1.0, 2.0, 0.5],
                                 [1.0, 2.0, 0.5, 3.0, 1.0/3.0],
                                 [1.0, 2.0, 0.5, 3.0, 1.0/3.0],
                                 [1.0, 2.0, 0.5, 3.0, 1.0/3.0],
                                 [1.0, 2.0, 0.5],
                                 [1.0, 2.0, 0.5]]
hyper_params["feature_map_shapes"] = [38, 19, 10, 5, 3, 1]
hyper_params["n_classes"] = 11+1
hyper_params["iou_threshold"] = 0.5
hyper_params["neg_pos_ratio"] = 3
hyper_params["loc_loss_alpha"] = 1
hyper_params["variances"] = [0.1, 0.2, 0.1, 0.2]
# hyper_params["scales"] = [0.1, 0.2, 0.37, 0.54, 0.71, 0.88, 1.05]
# hyper_params["mean_color"] = [123, 117, 104]


# training hyperparameter setting
IMAGE_SIZE = (300, 300)
EPOCHS = 2
BATCH_SIZE =32

### Dataset Generation
data_shapes = preprocessing.get_data_shapes()
padding_values = preprocessing.get_padding_values()

train_ds, train_len = preprocessing.ds_generator('dataset/train.csv', image_size=IMAGE_SIZE, augmentation=augmentation.apply)
val_ds, val_len = preprocessing.ds_generator('dataset/validation.csv', image_size=IMAGE_SIZE)

train_ds = train_ds.shuffle(100).padded_batch(BATCH_SIZE, padded_shapes=data_shapes, padding_values=padding_values)
val_ds = val_ds.padded_batch(BATCH_SIZE, padded_shapes=data_shapes, padding_values=padding_values)


### Modeling
ssd_model = get_model(hyper_params)
ssd_custom_losses = CustomLoss(hyper_params["neg_pos_ratio"], hyper_params["loc_loss_alpha"])
ssd_model.compile(optimizer=Adam(learning_rate=1e-3),
                  loss=[ssd_custom_losses.loc_loss_fn, ssd_custom_losses.conf_loss_fn])
init_model(ssd_model)

# We calculate prior boxes for one time and use it for all operations because of the all images are the same sizes
prior_boxes = generate_prior_boxes(hyper_params["feature_map_shapes"], hyper_params["aspect_ratios"])
ssd_train_feed = generator(train_ds, prior_boxes, hyper_params)
ssd_val_feed = generator(val_ds, prior_boxes, hyper_params)

### Callback setting
ssd_log_path = os.path.join("logs", "VGG", datetime.now().strftime("%Y%m%d-%H%M%S"))
ssd_model_path = os.path.join("trained", datetime.now().strftime("%Y%m%d-%H%M%S"))
model_name = os.path.join(ssd_model_path, "ssd300_vgg_weights.h5")

checkpoint_callback = ModelCheckpoint(model_name, monitor="val_loss", save_best_only=True, save_weights_only=True)
tensorboard_callback = TensorBoard(log_dir=ssd_log_path)
learning_rate_callback = LearningRateScheduler(scheduler, verbose=0)

### Model fitting
step_size_train = get_step_size(train_len, BATCH_SIZE)
step_size_val = get_step_size(val_len, BATCH_SIZE)

os.makedirs(ssd_model_path, exist_ok=True)
os.makedirs(ssd_log_path, exist_ok=True)

ssd_model.fit(ssd_train_feed,
              steps_per_epoch=step_size_train,
              validation_data=ssd_val_feed,
              validation_steps=step_size_val,
              epochs=EPOCHS,
              callbacks=[checkpoint_callback, tensorboard_callback, learning_rate_callback]
              )



