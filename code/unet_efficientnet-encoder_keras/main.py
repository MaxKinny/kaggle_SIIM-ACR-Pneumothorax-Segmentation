import pandas as pd
import glob
import seaborn as sns
import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split
import keras
import keras.backend as K
from keras.losses import binary_crossentropy
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
from keras.layers import Input, Dense, MaxPooling2D, Conv2DTranspose, concatenate, Multiply, Dropout, Add, Conv2D, BatchNormalization, LeakyReLU
from efficientnet import EfficientNetB5 as EfficientNetBackbone
from keras.models import Model
import cv2
from matplotlib import pyplot as plt
import shutil
import os
import tensorflow as tf
import gc
from tqdm import tqdm_notebook
import sys
sys.path.insert(0, '../')
sys.path.insert(0, './Keras-NASNet-master/')
from mask_functions import rle2mask, mask2rle
from nasnet import NASNetLarge
##########Data Augmentation############################################
from albumentations import (
    Compose, HorizontalFlip, CLAHE, HueSaturationValue,
    RandomBrightness, RandomContrast, RandomGamma,OneOf,
    ToFloat, ShiftScaleRotate,GridDistortion, ElasticTransform, JpegCompression, HueSaturationValue,
    RGBShift, RandomBrightness, RandomContrast, Blur, MotionBlur, MedianBlur, GaussNoise,CenterCrop,
    IAAAdditiveGaussianNoise,GaussNoise,OpticalDistortion
)

AUGMENTATIONS_TRAIN = Compose([
    HorizontalFlip(p=0.5),
    OneOf([
        RandomContrast(),
        RandomGamma(),
        RandomBrightness(),
         ], p=0.3),
    OneOf([
        ElasticTransform(alpha=120, sigma=120 * 0.05, alpha_affine=120 * 0.03),
        GridDistortion(),
        OpticalDistortion(distort_limit=2, shift_limit=0.5),
        ], p=0.3),
    ToFloat(max_value=1)
    ], p=1)


AUGMENTATIONS_TEST = Compose([
    ToFloat(max_value=1)
    ], p=1)
##########Data Augmentation############################################


def UEfficientNet(input_shape=(None, None, 3), dropout_rate=0.1):
    # backbone = EfficientNetBackbone(weights='imagenet',
    #                           include_top=False,
    #                           input_shape=input_shape)
    backbone = NASNetLarge(input_shape=input_shape, dropout=dropout_rate, include_top=False)
    backbone.summary()
    input = backbone.input
    start_neurons = 16

    conv4 = backbone.layers[342].output
    conv4 = LeakyReLU(alpha=0.1)(conv4)
    pool4 = MaxPooling2D((2, 2))(conv4)
    pool4 = Dropout(dropout_rate)(pool4)

    # Middle
    convm = Conv2D(start_neurons * 32, (3, 3), activation=None, padding="same")(pool4)
    convm = residual_block(convm, start_neurons * 32)
    convm = residual_block(convm, start_neurons * 32)
    convm = LeakyReLU(alpha=0.1)(convm)

    deconv4 = Conv2DTranspose(start_neurons * 16, (3, 3), strides=(2, 2), padding="same")(convm)
    uconv4 = concatenate([deconv4, conv4])
    uconv4 = Dropout(dropout_rate)(uconv4)

    uconv4 = Conv2D(start_neurons * 16, (3, 3), activation=None, padding="same")(uconv4)
    uconv4 = residual_block(uconv4, start_neurons * 16)
    uconv4 = residual_block(uconv4, start_neurons * 16)
    uconv4 = LeakyReLU(alpha=0.1)(uconv4)

    deconv3 = Conv2DTranspose(start_neurons * 8, (3, 3), strides=(2, 2), padding="same")(uconv4)
    conv3 = backbone.layers[154].output
    uconv3 = concatenate([deconv3, conv3])  #report error!
    uconv3 = Dropout(dropout_rate)(uconv3)

    uconv3 = Conv2D(start_neurons * 8, (3, 3), activation=None, padding="same")(uconv3)
    uconv3 = residual_block(uconv3, start_neurons * 8)
    uconv3 = residual_block(uconv3, start_neurons * 8)
    uconv3 = LeakyReLU(alpha=0.1)(uconv3)

    deconv2 = Conv2DTranspose(start_neurons * 4, (3, 3), strides=(2, 2), padding="same")(uconv3)
    conv2 = backbone.layers[92].output
    uconv2 = concatenate([deconv2, conv2])

    uconv2 = Dropout(0.1)(uconv2)
    uconv2 = Conv2D(start_neurons * 4, (3, 3), activation=None, padding="same")(uconv2)
    uconv2 = residual_block(uconv2, start_neurons * 4)
    uconv2 = residual_block(uconv2, start_neurons * 4)
    uconv2 = LeakyReLU(alpha=0.1)(uconv2)

    deconv1 = Conv2DTranspose(start_neurons * 2, (3, 3), strides=(2, 2), padding="same")(uconv2)
    conv1 = backbone.layers[30].output
    uconv1 = concatenate([deconv1, conv1])

    uconv1 = Dropout(0.1)(uconv1)
    uconv1 = Conv2D(start_neurons * 2, (3, 3), activation=None, padding="same")(uconv1)
    uconv1 = residual_block(uconv1, start_neurons * 2)
    uconv1 = residual_block(uconv1, start_neurons * 2)
    uconv1 = LeakyReLU(alpha=0.1)(uconv1)

    uconv0 = Conv2DTranspose(start_neurons * 1, (3, 3), strides=(2, 2), padding="same")(uconv1)
    uconv0 = Dropout(0.1)(uconv0)
    uconv0 = Conv2D(start_neurons * 1, (3, 3), activation=None, padding="same")(uconv0)
    uconv0 = residual_block(uconv0, start_neurons * 1)
    uconv0 = residual_block(uconv0, start_neurons * 1)
    uconv0 = LeakyReLU(alpha=0.1)(uconv0)

    uconv0 = Dropout(dropout_rate / 2)(uconv0)
    output_layer = Conv2D(1, (1, 1), padding="same", activation="sigmoid")(uconv0)

    model = Model(input, output_layer)
    model.name = 'u-xception'

    return model


def convolution_block(x, filters, size, strides=(1,1), padding='same', activation=True):
    x = Conv2D(filters, size, strides=strides, padding=padding)(x)
    x = BatchNormalization()(x)
    if activation == True:
        x = LeakyReLU(alpha=0.1)(x)
    return x


def residual_block(blockInput, num_filters=16):
    x = LeakyReLU(alpha=0.1)(blockInput)
    x = BatchNormalization()(x)
    blockInput = BatchNormalization()(blockInput)
    x = convolution_block(x, num_filters, (3,3) )
    x = convolution_block(x, num_filters, (3,3), activation=False)
    x = Add()([x, blockInput])
    return x


def dice_coef(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred = K.cast(y_pred, 'float32')
    y_pred_f = K.cast(K.greater(K.flatten(y_pred), 0.5), 'float32')
    intersection = y_true_f * y_pred_f
    score = 2. * K.sum(intersection) / (K.sum(y_true_f) + K.sum(y_pred_f))
    return score


def dice_loss(y_true, y_pred):
    smooth = 1.
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = y_true_f * y_pred_f
    score = (2. * K.sum(intersection) + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)
    return 1. - score


def bce_dice_loss(y_true, y_pred):
    return binary_crossentropy(y_true, y_pred) + dice_loss(y_true, y_pred)


def bce_logdice_loss(y_true, y_pred):
    return binary_crossentropy(y_true, y_pred) - K.log(1. - dice_loss(y_true, y_pred))


def get_iou_vector(A, B):
    # Numpy version
    batch_size = A.shape[0]
    metric = 0.0
    for batch in range(batch_size):
        t, p = A[batch], B[batch]
        true = np.sum(t)
        pred = np.sum(p)

        # deal with empty mask first
        if true == 0:
            metric += (pred == 0)
            continue

        # non empty mask case.  Union is never empty
        # hence it is safe to divide by its number of pixels
        intersection = np.sum(t * p)
        union = true + pred - intersection
        iou = intersection / union

        # iou metrric is a stepwise approximation of the real iou over 0.5
        iou = np.floor(max(0, (iou - 0.45) * 20)) / 10

        metric += iou

    # teake the average over all images in batch
    metric /= batch_size
    return metric


def my_iou_metric(label, pred):
    # Tensorflow version
    return tf.py_func(get_iou_vector, [label, pred > 0.5], tf.float64)


class SnapshotCallbackBuilder:
    def __init__(self, nb_epochs, nb_snapshots, init_lr=0.1):
        self.T = nb_epochs
        self.M = nb_snapshots
        self.alpha_zero = init_lr

    def get_callbacks(self, model_prefix='Model'):

        callback_list = [
            ModelCheckpoint("./keras.model", monitor='val_my_iou_metric',
                                   mode='max', save_best_only=True, verbose=1),
            swa,
            LearningRateScheduler(schedule=self._cosine_anneal_schedule)
        ]

        return callback_list

    def _cosine_anneal_schedule(self, t):
        cos_inner = np.pi * (t % (self.T // self.M))  # t - 1 is used when t has 1-based indexing.
        cos_inner /= self.T // self.M
        cos_out = np.cos(cos_inner) + 1
        return float(self.alpha_zero / 2 * cos_out)


class DataGenerator(keras.utils.Sequence):
    'Generates data for Keras'

    def __init__(self, train_im_path='./keras_im_train', train_mask_path='./keras_mask_train',
                 augmentations=None, batch_size=16, img_size=256, n_channels=3, shuffle=True):
        'Initialization'
        self.batch_size = batch_size
        self.train_im_paths = glob.glob(train_im_path + '/*')

        self.train_im_path = train_im_path
        self.train_mask_path = train_mask_path

        self.img_size = img_size

        self.n_channels = n_channels
        self.shuffle = shuffle
        self.augment = augmentations
        self.on_epoch_end()

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.ceil(len(self.train_im_paths) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index * self.batch_size:min((index + 1) * self.batch_size, len(self.train_im_paths))]

        # Find list of IDs
        list_IDs_im = [self.train_im_paths[k] for k in indexes]

        # Generate data
        X, y = self.data_generation(list_IDs_im)

        if self.augment is None:
            return X, np.array(y) / 255
        else:
            im, mask = [], []
            for x, y in zip(X, y):
                augmented = self.augment(image=x, mask=y)
                im.append(augmented['image'])
                mask.append(augmented['mask'])
            return np.array(im), np.array(mask) / 255

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.train_im_paths))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def data_generation(self, list_IDs_im):
        'Generates data containing batch_size samples'  # X : (n_samples, *dim, n_channels)
        # Initialization
        X = np.empty((len(list_IDs_im), self.img_size, self.img_size, self.n_channels))
        y = np.empty((len(list_IDs_im), self.img_size, self.img_size, 1))

        # Generate data
        for i, im_path in enumerate(list_IDs_im):

            im = np.array(Image.open(im_path))
            mask_path = im_path.replace(self.train_im_path, self.train_mask_path)

            mask = np.array(Image.open(mask_path))

            if len(im.shape) == 2:
                im = np.repeat(im[..., None], 3, 2)

            #             # Resize sample
            X[i,] = cv2.resize(im, (self.img_size, self.img_size))

            # Store class
            y[i,] = cv2.resize(mask, (self.img_size, self.img_size))[..., np.newaxis]
            y[y > 0] = 255

        return np.uint8(X), np.uint8(y)


class SWA(keras.callbacks.Callback):

    def __init__(self, filepath, swa_epoch):
        super(SWA, self).__init__()
        self.filepath = filepath
        self.swa_epoch = swa_epoch

    def on_train_begin(self, logs=None):
        self.nb_epoch = self.params['epochs']
        print('Stochastic weight averaging selected for last {} epochs.'
              .format(self.nb_epoch - self.swa_epoch))

    def on_epoch_end(self, epoch, logs=None):

        if epoch == self.swa_epoch:
            self.swa_weights = self.model.get_weights()

        elif epoch > self.swa_epoch:
            for i in range(len(self.swa_weights)):
                self.swa_weights[i] = (self.swa_weights[i] *
                                       (epoch - self.swa_epoch) + self.model.get_weights()[i]) / (
                                                  (epoch - self.swa_epoch) + 1)

        else:
            pass

    def on_train_end(self, logs=None):
        self.model.set_weights(self.swa_weights)
        print('Final model parameters set to stochastic weight average.')
        self.model.save_weights(self.filepath)
        print('Final stochastic averaged weights saved to file.')


def predict_result(model,validation_generator,img_size):
    # TBD predict both orginal and reflect x
    preds_test1 = model.predict_generator(validation_generator).reshape(-1, img_size, img_size)
    return preds_test1


# src: https://www.kaggle.com/aglotero/another-iou-metric
def iou_metric(y_true_in, y_pred_in, print_table=False):
    labels = y_true_in
    y_pred = y_pred_in

    true_objects = 2
    pred_objects = 2

    intersection = np.histogram2d(labels.flatten(), y_pred.flatten(), bins=(true_objects, pred_objects))[0]

    # Compute areas (needed for finding the union between all objects)
    area_true = np.histogram(labels, bins=true_objects)[0]
    area_pred = np.histogram(y_pred, bins=pred_objects)[0]
    area_true = np.expand_dims(area_true, -1)
    area_pred = np.expand_dims(area_pred, 0)

    # Compute union
    union = area_true + area_pred - intersection

    # Exclude background from the analysis
    intersection = intersection[1:, 1:]
    union = union[1:, 1:]
    union[union == 0] = 1e-9

    # Compute the intersection over union
    iou = intersection / union

    # Precision helper function
    def precision_at(threshold, iou):
        matches = iou > threshold
        true_positives = np.sum(matches, axis=1) == 1  # Correct objects
        false_positives = np.sum(matches, axis=0) == 0  # Missed objects
        false_negatives = np.sum(matches, axis=1) == 0  # Extra objects
        tp, fp, fn = np.sum(true_positives), np.sum(false_positives), np.sum(false_negatives)
        return tp, fp, fn

    # Loop over IoU thresholds
    prec = []
    if print_table:
        print("Thresh\tTP\tFP\tFN\tPrec.")
    for t in np.arange(0.5, 1.0, 0.05):
        tp, fp, fn = precision_at(t, iou)
        if (tp + fp + fn) > 0:
            p = tp / (tp + fp + fn)
        else:
            p = 0
        if print_table:
            print("{:1.3f}\t{}\t{}\t{}\t{:1.3f}".format(t, tp, fp, fn, p))
        prec.append(p)

    if print_table:
        print("AP\t-\t-\t-\t{:1.3f}".format(np.mean(prec)))
    return np.mean(prec)


def iou_metric_batch(y_true_in, y_pred_in):
    batch_size = y_true_in.shape[0]
    metric = []
    for batch in range(batch_size):
        value = iou_metric(y_true_in[batch], y_pred_in[batch])
        metric.append(value)
    return np.mean(metric)


if __name__ == '__main__':
    train_im_path, train_mask_path = './keras_im_train', './keras_mask_train'
    h, w, batch_size = 256, 256, 16
    img_size = 256
    val_im_path, val_mask_path = './keras_im_val', './keras_mask_val'
    # percentage of mask
    all_mask_fn = glob.glob('../../data/png/256/train/mask/*')
    mask_df = pd.DataFrame()
    mask_df['file_names'] = all_mask_fn
    mask_df['mask_percentage'] = 0
    mask_df.set_index('file_names', inplace=True)
    for fn in all_mask_fn:
        mask_df.loc[fn, 'mask_percentage'] = np.array(Image.open(fn)).sum() / (
                    256 * 256 * 255)  # 255 is bcz img range is 255

    mask_df.reset_index(inplace=True)
    sns.distplot(mask_df.mask_percentage)
    mask_df['labels'] = 0
    mask_df.loc[mask_df.mask_percentage > 0, 'labels'] = 1

    # splitting
    all_train_fn = glob.glob('../../data/png/256/train/picture/*')
    total_samples = len(all_train_fn)
    idx = np.arange(total_samples)
    train_fn, val_fn = train_test_split(all_train_fn, stratify=mask_df.labels, test_size=0.1, random_state=10)

    print('No. of train files:', len(train_fn))
    print('No. of val files:', len(val_fn))

    masks_train_fn = [fn.replace('picture', 'mask') for fn in train_fn]
    masks_val_fn = [fn.replace('picture', 'mask') for fn in val_fn]

    # move splitted files
    file1 = glob.glob(train_im_path+'/*.png')
    file2 = glob.glob(train_mask_path+'/*.png')
    file3 = glob.glob(val_mask_path + '/*.png')
    file4 = glob.glob(val_im_path + '/*.png')
    [os.remove(fp) for fp in file1]
    [os.remove(fp) for fp in file2]
    [os.remove(fp) for fp in file3]
    [os.remove(fp) for fp in file4]
    [shutil.copy(fp, train_im_path+'/') for fp in train_fn]
    [shutil.copy(fp, train_mask_path+'/') for fp in masks_train_fn]
    [shutil.copy(fp, val_mask_path+'/') for fp in masks_val_fn]
    [shutil.copy(fp, val_im_path+'/') for fp in val_fn]

    # Train Set Images with Masks
    # a = DataGenerator(batch_size=64, shuffle=False, train_im_path=train_im_path, train_mask_path=train_mask_path)
    # images, masks = a.__getitem__(0)
    # max_images = 64
    # grid_width = 16
    # grid_height = int(max_images / grid_width)
    # fig, axs = plt.subplots(grid_height, grid_width, figsize=(grid_width, grid_height))
    #
    # for i, (im, mask) in enumerate(zip(images, masks)):
    #     ax = axs[int(i / grid_width), i % grid_width]
    #     ax.imshow(im.squeeze(), cmap="bone")
    #     ax.imshow(mask.squeeze(), alpha=0.5, cmap="Reds")
    #     ax.axis('off')
    # plt.suptitle("Chest X-rays, Red: Pneumothorax.")
    # Augmentations
    # a = DataGenerator(batch_size=64, augmentations=AUGMENTATIONS_TRAIN, shuffle=False)
    # images, masks = a.__getitem__(0)
    # max_images = 64
    # grid_width = 16
    # grid_height = int(max_images / grid_width)
    # # fig, axs = plt.subplots(grid_height, grid_width, figsize=(grid_width, grid_height))
    #
    # for i, (im, mask) in enumerate(zip(images, masks)):
    #     ax = axs[int(i / grid_width), i % grid_width]
    #     ax.imshow(im[:, :, 0], cmap="bone")
    #     ax.imshow(mask.squeeze(), alpha=0.5, cmap="Reds")
    #     ax.axis('off')
    # plt.suptitle("Chest X-rays, Red: Pneumothorax.")


    #################training###############################
    K.clear_session()
    model = UEfficientNet(input_shape=(img_size, img_size, 3), dropout_rate=0.25)
    model.compile(loss=bce_dice_loss, optimizer='adam', metrics=[my_iou_metric])
    epochs = 60
    snapshot = SnapshotCallbackBuilder(nb_epochs=epochs, nb_snapshots=1, init_lr=1e-3)
    batch_size = 8
    swa = SWA('./keras_swa.model', 67)
    valid_im_path, valid_mask_path = './keras_im_val', './keras_mask_val'
    # Generators
    training_generator = DataGenerator(augmentations=AUGMENTATIONS_TRAIN,
                                       img_size=img_size,
                                       batch_size=batch_size)
    validation_generator = DataGenerator(train_im_path=valid_im_path,
                                         train_mask_path=valid_mask_path,
                                         augmentations=AUGMENTATIONS_TEST,
                                         img_size=img_size,
                                         batch_size=batch_size)
    history = model.fit_generator(generator=training_generator,
                                  validation_data=validation_generator,
                                  use_multiprocessing=False,
                                  epochs=epochs,
                                  verbose=1,
                                  callbacks=snapshot.get_callbacks())
    # plot training states
    plt.figure(figsize=(16, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['my_iou_metric'][1:])
    plt.plot(history.history['val_my_iou_metric'][1:])
    plt.ylabel('iou')
    plt.xlabel('epoch')
    plt.legend(['train', 'Validation'], loc='upper left')

    plt.title('model IOU')

    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'][1:])
    plt.plot(history.history['val_loss'][1:])
    plt.ylabel('val_loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'Validation'], loc='upper left')
    plt.title('model loss')
    gc.collect()
    # Load best model or swa model if not available
    try:
        print('using swa weight model')
        model.load_weights('./keras_swa.model')
    except Exception as e:
        print(e)
        model.load_weights('./keras.model')

    # Predict the validation set to do a sanity check
    validation_generator = DataGenerator(train_im_path=valid_im_path,
                                         train_mask_path=valid_mask_path,
                                         augmentations=AUGMENTATIONS_TEST,
                                         img_size=img_size,
                                         shuffle=False)
    preds_valid = predict_result(model, validation_generator, img_size)
    valid_fn = glob.glob('./keras_mask_val/*')
    y_valid_ori = np.array([cv2.resize(np.array(Image.open(fn)), (img_size, img_size)) for fn in valid_fn])
    assert y_valid_ori.shape == preds_valid.shape
    # Plot some predictions for validation set images
    threshold_best = 0.5
    max_images = 64
    grid_width = 16
    grid_height = int(max_images / grid_width)
    fig, axs = plt.subplots(grid_height, grid_width, figsize=(grid_width, grid_height))

    validation_generator = DataGenerator(train_im_path=valid_im_path,
                                         train_mask_path=valid_mask_path, augmentations=AUGMENTATIONS_TEST,
                                         img_size=img_size, batch_size=64, shuffle=False)

    images, masks = validation_generator.__getitem__(0)
    i = 0
    for i, (im, mask) in enumerate(zip(images, masks)):
        print("**************", i)
        pred = preds_valid[i]
        ax = axs[int(i / grid_width), i % grid_width]
        ax.imshow(im[..., 0], cmap="bone")
        ax.imshow(mask.squeeze(), alpha=0.5, cmap="Reds")
        ax.imshow(np.array(np.round(pred > threshold_best), dtype=np.float32), alpha=0.5, cmap="Greens")
        ax.axis('off')
    plt.suptitle("Green:Prediction , Red: Pneumothorax.")
    ## Scoring for last model
    thresholds = np.linspace(0.2, 0.9, 31)
    ious = np.array(
        [iou_metric_batch(y_valid_ori, np.int32(preds_valid > threshold)) for threshold in tqdm_notebook(thresholds)])
    threshold_best_index = np.argmax(ious)
    iou_best = ious[threshold_best_index]
    threshold_best = thresholds[threshold_best_index]

    plt.plot(thresholds, ious)
    plt.plot(threshold_best, iou_best, "xr", label="Best threshold")
    plt.xlabel("Threshold")
    plt.ylabel("IoU")
    plt.title("Threshold vs IoU ({}, {})".format(threshold_best, iou_best))
    plt.legend()

    max_images = 64
    grid_width = 16
    grid_height = int(max_images / grid_width)
    fig, axs = plt.subplots(grid_height, grid_width, figsize=(grid_width, grid_height))

    validation_generator = DataGenerator(train_im_path=valid_im_path,
                                         train_mask_path=valid_mask_path, augmentations=AUGMENTATIONS_TEST,
                                         img_size=img_size, batch_size=64, shuffle=False)

    images, masks = validation_generator.__getitem__(0)
    for i, (im, mask) in enumerate(zip(images, masks)):
        pred = preds_valid[i]
        ax = axs[int(i / grid_width), i % grid_width]
        ax.imshow(im[..., 0], cmap="bone")
        ax.imshow(mask.squeeze(), alpha=0.5, cmap="Reds")
        ax.imshow(np.array(np.round(pred > threshold_best), dtype=np.float32), alpha=0.5, cmap="Greens")
        ax.axis('off')
    plt.suptitle("Green:Prediction , Red: Pneumothorax.")

    #################Test set prediction###############################
    test_fn = glob.glob('../../data/png/256/test/*')
    x_test = [cv2.resize(np.array(Image.open(fn)), (img_size, img_size)) for fn in test_fn]
    x_test = np.array(x_test)
    x_test = np.array([np.repeat(im[..., None], 3, 2) for im in x_test])
    print(x_test.shape)
    preds_test = model.predict(x_test, batch_size=batch_size)
    # del x_test; gc.collect()
    # Some Test Set Predictions
    max_images = 64
    grid_width = 16
    grid_height = int(max_images / grid_width)
    fig, axs = plt.subplots(grid_height, grid_width, figsize=(grid_width, grid_height))
    # for i, idx in enumerate(index_val[:max_images]):
    for i, idx in enumerate(test_fn[:max_images]):
        img = x_test[i]
        pred = preds_test[i].squeeze()
        ax = axs[int(i / grid_width), i % grid_width]
        ax.imshow(img, cmap="Greys")
        ax.imshow(np.array(np.round(pred > threshold_best), dtype=np.float32), alpha=0.5, cmap="Reds")
        ax.axis('off')
    # Generate rle encodings (images are first converted to the original size)
    rles = []
    i, max_img = 1, 10
    plt.figure(figsize=(16, 4))
    i = 0
    for p in tqdm_notebook(preds_test):
        i += 1
        print("*****", i)
        p = p.squeeze()
        im = cv2.resize(p, (1024, 1024))
        im = im > threshold_best
        #     zero out the smaller regions.
        if im.sum() < 1024 * 2:
            im[:] = 0
        im = (im.T * 255).astype(np.uint8)
        rles.append(mask2rle(im, 1024, 1024))
        i += 1
        if i < max_img:
            plt.subplot(1, max_img, i)
            plt.imshow(im)
            plt.axis('off')
    ids = [o.split('/')[-1][:-4] for o in test_fn]
    sub_df = pd.DataFrame({'ImageId': ids, 'EncodedPixels': rles})
    sub_df.loc[sub_df.EncodedPixels == '', 'EncodedPixels'] = '-1'
    sub_df.to_csv('submission.csv', index=False)
    sub_df.head()
    sub_df.tail(10)
