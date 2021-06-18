import os
from utils import bbox_utils, preprocessing, train_utils, eval_utils, drawing_utils
from models.decoder import get_decoder_model
from models.ssd_vgg16 import get_model, init_model


# hyper parameters
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
hyper_params["variances"] = [0.1, 0.1, 0.2, 0.2]

# data setting
labels = os.listdir('dataset')[:11]
labels = ['bg'] + labels
IMAGE_SIZE = hyper_params['image_size']
BATCH_SIZE = 32
backbone = 'vgg16'

data_shapes = preprocessing.get_data_shapes()
padding_values = preprocessing.get_padding_values()

test_ds, test_len = preprocessing.ds_generator('dataset/test.csv', image_size=IMAGE_SIZE, augmentation=False)
test_ds = test_ds.padded_batch(BATCH_SIZE, padded_shapes=data_shapes, padding_values=padding_values)

###

ssd_model = get_model(hyper_params)
ssd_model_path = os.path.join("trained", os.listdir("trained")[-1], "ssd300_vgg_weights.h5")
ssd_model.load_weights(ssd_model_path)

prior_boxes = bbox_utils.generate_prior_boxes(hyper_params["feature_map_shapes"], hyper_params["aspect_ratios"])
ssd_decoder_model = get_decoder_model(ssd_model, prior_boxes, hyper_params)

step_size = train_utils.get_step_size(test_len, BATCH_SIZE)
pred_bboxes, pred_labels, pred_scores = ssd_decoder_model.predict(test_ds, steps=step_size, verbose=1)

###

evaluate=False
if evaluate:
    eval_utils.evaluate_predictions(test_ds, pred_bboxes, pred_labels, pred_scores, labels, BATCH_SIZE)
else:
    drawing_utils.draw_predictions(test_ds, pred_bboxes, pred_labels, pred_scores, labels, BATCH_SIZE)