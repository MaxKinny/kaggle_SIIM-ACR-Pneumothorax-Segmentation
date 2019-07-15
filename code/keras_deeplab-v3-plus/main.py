import numpy as np
import pandas as pd
import glob
from tqdm import tqdm_notebook
from tqdm import tqdm_notebook as tqdm
from matplotlib import pyplot as plt
import sys
import pydicom
from mask_functions import rle2mask, mask2rle
import keras as K
from keras.layers import Input, Conv2D, MaxPooling2D, Conv2DTranspose, concatenate
from keras.models import Model
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
from keras.callbacks import TensorBoard
from model import Deeplabv3
from keras import backend as K
from PIL import Image
import os

from keras.utils import multi_gpu_utils


def show_dcm_info(dataset):
    print("Filename.........:", file_path)
    print("Storage type.....:", dataset.SOPClassUID)
    print()

    pat_name = dataset.PatientName
    display_name = pat_name.family_name + ", " + pat_name.given_name
    print("Patient's name......:", display_name)
    print("Patient id..........:", dataset.PatientID)
    print("Patient's Age.......:", dataset.PatientAge)
    print("Patient's Sex.......:", dataset.PatientSex)
    print("Modality............:", dataset.Modality)
    print("Body Part Examined..:", dataset.BodyPartExamined)
    print("View Position.......:", dataset.ViewPosition)

    if 'PixelData' in dataset:
        rows = int(dataset.Rows)
        cols = int(dataset.Columns)
        print("Image size.......: {rows:d} x {cols:d}, {size:d} bytes".format(
            rows=rows, cols=cols, size=len(dataset.PixelData)))
        if 'PixelSpacing' in dataset:
            print("Pixel spacing....:", dataset.PixelSpacing)


def plot_pixel_array(dataset, figsize=(10, 10)):
    plt.figure(figsize=figsize)
    plt.imshow(dataset.pixel_array, cmap=plt.cm.bone)
    plt.show()


def dice_coef(y_true, y_pred, smooth=1):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)


def dice_coef_loss(y_true, y_pred):
    return -dice_coef(y_true, y_pred)


def Unet(im_chan):
    inputs = Input((None, None, im_chan))

    c1 = Conv2D(8, (3, 3), activation='relu', padding='same')(inputs)
    c1 = Conv2D(8, (3, 3), activation='relu', padding='same')(c1)
    p1 = MaxPooling2D((2, 2))(c1)

    c2 = Conv2D(16, (3, 3), activation='relu', padding='same')(p1)
    c2 = Conv2D(16, (3, 3), activation='relu', padding='same')(c2)
    p2 = MaxPooling2D((2, 2))(c2)

    c3 = Conv2D(32, (3, 3), activation='relu', padding='same')(p2)
    c3 = Conv2D(32, (3, 3), activation='relu', padding='same')(c3)
    p3 = MaxPooling2D((2, 2))(c3)

    c4 = Conv2D(64, (3, 3), activation='relu', padding='same')(p3)
    c4 = Conv2D(64, (3, 3), activation='relu', padding='same')(c4)
    p4 = MaxPooling2D(pool_size=(2, 2))(c4)

    c5 = Conv2D(64, (3, 3), activation='relu', padding='same')(p4)
    c5 = Conv2D(64, (3, 3), activation='relu', padding='same')(c5)
    p5 = MaxPooling2D(pool_size=(2, 2))(c5)

    c55 = Conv2D(128, (3, 3), activation='relu', padding='same')(p5)
    c55 = Conv2D(128, (3, 3), activation='relu', padding='same')(c55)

    u6 = Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(c55)
    u6 = concatenate([u6, c5])
    c6 = Conv2D(64, (3, 3), activation='relu', padding='same')(u6)
    c6 = Conv2D(64, (3, 3), activation='relu', padding='same')(c6)

    u71 = Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same')(c6)
    u71 = concatenate([u71, c4])
    c71 = Conv2D(32, (3, 3), activation='relu', padding='same')(u71)
    c61 = Conv2D(32, (3, 3), activation='relu', padding='same')(c71)

    u7 = Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same')(c61)
    u7 = concatenate([u7, c3])
    c7 = Conv2D(32, (3, 3), activation='relu', padding='same')(u7)
    c7 = Conv2D(32, (3, 3), activation='relu', padding='same')(c7)

    u8 = Conv2DTranspose(16, (2, 2), strides=(2, 2), padding='same')(c7)
    u8 = concatenate([u8, c2])
    c8 = Conv2D(16, (3, 3), activation='relu', padding='same')(u8)
    c8 = Conv2D(16, (3, 3), activation='relu', padding='same')(c8)

    u9 = Conv2DTranspose(8, (2, 2), strides=(2, 2), padding='same')(c8)
    u9 = concatenate([u9, c1], axis=3)
    c9 = Conv2D(8, (3, 3), activation='relu', padding='same')(u9)
    c9 = Conv2D(8, (3, 3), activation='relu', padding='same')(c9)

    outputs = Conv2D(1, (1, 1), activation='sigmoid')(c9)

    return Model(inputs=[inputs], outputs=[outputs])


if __name__ == '__main__':
    basestr = 'deeplabv3P'
    im_height = 1024
    im_width = 1024
    im_chan = 1
    #################################
    #########Dataset#################
    #################################
    # Load Full Dataset
    train_glob = '../../data/dicom/dicom-images-train/*/*/*.dcm'
    test_glob = '../../data/dicom/dicom-images-test/*/*/*.dcm'
    train_fns = sorted(glob.glob(train_glob))[:5000]
    test_fns = sorted(glob.glob(test_glob))[:5000]
    # for file_path in train_fns:
    #     dataset = pydicom.dcmread(file_path)
    #     show_dcm_info(dataset)
    #     plot_pixel_array(dataset)
    #     break  # Comment this out to see all
    df_full = pd.read_csv('../dicom_data/train-rle.csv', index_col='ImageId')
    # Get train images and masks
    X_train = np.zeros((len(train_fns), im_height, im_width, im_chan), dtype=np.uint8)
    Y_train = np.zeros((len(train_fns), im_height, im_width, im_chan), dtype=np.bool)
    print('Getting train images and masks ... ')
    sys.stdout.flush()
    for n, _id in tqdm_notebook(enumerate(train_fns), total=len(train_fns)):
        dataset = pydicom.read_file(_id)
        X_train[n] = np.expand_dims(dataset.pixel_array, axis=2)
        try:
            if '-1' in df_full.loc[_id.split('/')[-1][:-4], ' EncodedPixels']:
                Y_train[n] = np.zeros((1024, 1024, 1))
            else:
                if type(df_full.loc[_id.split('/')[-1][:-4], ' EncodedPixels']) == str:
                    Y_train[n] = np.expand_dims(
                        rle2mask(df_full.loc[_id.split('/')[-1][:-4], ' EncodedPixels'], 1024, 1024), axis=2)
                    # print("****", np.unique(rle2mask(df_full.loc[_id.split('/')[-1][:-4], ' EncodedPixels'], 1024, 1024)))
                    # plt.imshow(rle2mask(df_full.loc[_id.split('/')[-1][:-4], ' EncodedPixels'], 1024, 1024))
                    # plt.imshow(rle2mask(rle, 1024, 1024).T)
                    # break
                else:
                    Y_train[n] = np.zeros((1024, 1024, 1))
                    for x in df_full.loc[_id.split('/')[-1][:-4], ' EncodedPixels']:
                        Y_train[n] = Y_train[n] + np.expand_dims(rle2mask(x, 1024, 1024), axis=2)
        except KeyError:
            print(f"Key {_id.split('/')[-1][:-4]} without mask, assuming healthy patient.")
            Y_train[n] = np.zeros((1024, 1024, 1))  # Assume missing masks are empty masks.
    print('Done!')
    # Show picture and mask
    # for i in range(0, 100):
    #     if True in np.unique(Y_train[i][:, :, 0]):
    #         print(np.unique(Y_train[i][:, :, 0]), i)
    #         break
    # i = 13
    # plt.imshow(X_train[i][:, :, 0], cmap=plt.cm.bone)
    # plt.figure()
    # plt.imshow(Y_train[i][:, :, 0])

    # Build Patches
    # Reshape to get non-overlapping patches.
    im_reshape_height = 128
    im_reshape_width = 128
    X_train = X_train.reshape((-1, im_reshape_height, im_reshape_width, 1))
    Y_train = Y_train.reshape((-1, im_reshape_height, im_reshape_width, 1))
    #################################
    #########Training Part###########
    #################################
    file_path = "vgg_face_" + basestr + ".h5"
    # for saving the checkpoint
    checkpoint = ModelCheckpoint(file_path, monitor='val_acc', verbose=1, save_best_only=True, mode='max')

    # adaptively change the learning rate
    reduce_on_plateau = ReduceLROnPlateau(monitor="val_acc", mode="max", factor=0.1, patience=20, verbose=1)

    tbCallBack = TensorBoard(log_dir='./logs/' + basestr,
                             histogram_freq=0,
                             write_graph=True,
                             write_images=True)

    callbacks_list = [checkpoint, reduce_on_plateau, tbCallBack]

    # model = Deeplabv3(weights=None, input_shape=(im_reshape_height, im_reshape_width, im_chan), classes=1)
    model = Unet(im_chan)
    # parallel_model = multi_gpu_utils(model, gpus=1)
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=[dice_coef, 'acc', 'mse'])
    model.summary()
    model.fit(X_train, Y_train, validation_split=.2, batch_size=1, epochs=1, callbacks=callbacks_list)

    #################################
    #########Testing Part############
    #################################
    # Get Test Data

    sample_df = pd.read_csv("../../sample_submission/sample_submission.csv")

    # this part was taken from @raddar's kernel: https://www.kaggle.com/raddar/better-sample-submission
    masks_ = sample_df.groupby('ImageId')['ImageId'].count().reset_index(name='N').ImageId.values
    # masks_ = masks_.loc[masks_.N > 1].ImageId.values
    ###
    sample_df = sample_df.drop_duplicates('ImageId', keep='last').reset_index(drop=True)
    sublist = []
    counter = 0
    threshold = 0.5
    for index, row in tqdm(sample_df.iterrows(), total=len(sample_df)):
        image_id = row['ImageId']
        print("*****", index)
        if image_id in masks_:
            img_path = os.path.join('../../data/png/1024/test', image_id + '.png')
            img = Image.open(img_path)
            width, height = img.size
            img = np.array(img)
            img = np.expand_dims(img, axis=2)
            img = np.expand_dims(img, axis=0)
            pred = model.predict(img)
            print(img_path)
            print(np.unique(pred))
            mask = 255 * (pred > threshold).astype(np.uint8).T
            if np.count_nonzero(mask) == 0:
                rle = " -1"
            else:
                rle = mask2rle(mask, width, height)
                # plt.imshow(rle2mask(rle, 1024, 1024))
        else:
            rle = " -1"
        sublist.append([image_id, rle])

    submission_df = pd.DataFrame(sublist, columns=sample_df.columns.values)
    submission_df.to_csv("submission.csv", index=False)
    print('Counter: ', counter)
    # # Generates labels using most basic setup.  Supports various image sizes.  Returns image labels in same format
    # # as original image.  Normalization matches MobileNetV2
    #
    # trained_image_width=512
    # mean_subtraction_value=127.5
    # image = np.array(Image.open('imgs/image1.jpg'))
    #
    # # resize to max dimension of images from training dataset
    # w, h, _ = image.shape
    # ratio = float(trained_image_width) / np.max([w, h])
    # resized_image = np.array(Image.fromarray(image.astype('uint8')).resize((int(ratio * h), int(ratio * w))))
    #
    # # apply normalization for trained dataset images
    # resized_image = (resized_image / mean_subtraction_value) - 1.
    #
    # # pad array to square image to match training images
    # pad_x = int(trained_image_width - resized_image.shape[0])
    # pad_y = int(trained_image_width - resized_image.shape[1])
    # resized_image = np.pad(resized_image, ((0, pad_x), (0, pad_y), (0, 0)), mode='constant')
    #
    # # make prediction
    # deeplab_model = Deeplabv3(backbone='xception')
    # res = deeplab_model.predict(np.expand_dims(resized_image, 0))
    # labels = np.argmax(res.squeeze(), -1)
    #
    # # remove padding and resize back to original image
    # if pad_x > 0:
    #     labels = labels[:-pad_x]
    # if pad_y > 0:
    #     labels = labels[:, :-pad_y]
    # labels = np.array(Image.fromarray(labels.astype('uint8')).resize((h, w)))
    #
    # plt.imshow(labels)
    # plt.waitforbuttonpress()
