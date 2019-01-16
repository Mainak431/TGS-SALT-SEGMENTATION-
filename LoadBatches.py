import numpy as np
import cv2
import glob
import itertools
import os
from PIL import Image

def getImageArr(path1, width, height,dp, imgNorm="divide", odering='channels_first'):
    try:
        img = Image.open(path1)
        std = dp['std']
        mean = dp['mean']
        path1 = os.path.basename(path1)
        path1 = os.path.splitext(path1)[0]
        depth = dp[path1] - mean
        depth /= std
        #print(depth)
        img = img.resize((width,height))
        img = np.array(img)
        img = img.astype(np.float32)
        img = img / 255.0
        if depth == 0:
            depth = 0.0000001
        img = img / depth
        '''for i in range(width) :
            for j in range(height):
                img[i][j] += depth
        '''
        img = img.reshape(width,height,3)
        #print(img.shape)
        return img
    except :
        return img


def getSegmentationArr(path1, nClasses, width, height):
    img = Image.open(path1)
    img = img.resize((width, height))
    img = img.convert('L')
    img = np.array(img)
    img = img.astype(np.float32)
    img = img / 255.0
    for i in range(width) :
        for j in range(height) :
            if img[i][j] < 0.5 :
                img[i][j] = 0
            else :
                img[i][j] = 1
    img = img.reshape(width, height, 1)
    return img


def imageSegmentationGenerator(images_path1, segs_path1, batch_size, n_classes, input_height, input_width, output_height,
                               output_width,depth_dict):
    assert images_path1[-1] == '/'
    assert segs_path1[-1] == '/'

    images = glob.glob(images_path1 + "*.jpg") + glob.glob(images_path1 + "*.png") + glob.glob(images_path1 + "*.jpeg")
    images.sort()
    segmentations = glob.glob(segs_path1 + "*.jpg") + glob.glob(segs_path1 + "*.png") + glob.glob(segs_path1 + "*.jpeg")
    segmentations.sort()

    assert len(images) == len(segmentations)
    for im, seg in zip(images, segmentations):
        assert (im.split('/')[-1].split(".")[0] == seg.split('/')[-1].split(".")[0])

    zipped = itertools.cycle(zip(images, segmentations))

    while True:
        X = []
        Y = []
        for _ in range(batch_size):
            im, seg = next(zipped)
            X.append(getImageArr(im,input_width, input_height,depth_dict))
            Y.append(getSegmentationArr(seg, n_classes, output_width, output_height))

        yield np.array(X), np.array(Y)

# import Models , LoadBatches
# G  = LoadBatches.imageSegmentationGenerator( "data/clothes_seg/prepped/images_prepped_train/" ,  "data/clothes_seg/prepped/annotations_prepped_train/" ,  1,  10 , 800 , 550 , 400 , 272   )
# G2  = LoadBatches.imageSegmentationGenerator( "data/clothes_seg/prepped/images_prepped_test/" ,  "data/clothes_seg/prepped/annotations_prepped_test/" ,  1,  10 , 800 , 550 , 400 , 272   )

# m = Models.VGGSegnet.VGGSegnet( 10  , use_vgg_weights=True ,  optimizer='adadelta' , input_image_size=( 800 , 550 )  )
# m.fit_generator( G , 512  , nb_epoch=10 )
