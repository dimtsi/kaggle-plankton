#!/usr/bin/env python
# coding: utf-8

# In[11]:


from PIL import Image, ImageOps, ImageFilter
import pickle


# In[12]:


train_images = pickle.load(open("pkl/train_images.pkl", "rb"))
test_images = pickle.load(open("pkl/test_images.pkl", "rb"))
classified_train_images = pickle.load(open("pkl/classified_train_images.pkl", "rb"))

# In[13]:


def pad_and_resize(im):
    desired_size = 64
    old_size = im.size  # old_size[0] is in (width, height) format
    ratio = float(desired_size)/max(old_size)
    new_size = tuple([int(x*ratio) for x in old_size])
    im = im.resize(new_size, Image.ANTIALIAS)
    new_im = Image.new("RGB", (desired_size, desired_size), "white")
    new_im.paste(im, ((desired_size-new_size[0])//2,
                        (desired_size-new_size[1])//2))
    # new_im = im.resize((desired_size, desired_size), Image.ANTIALIAS)
    return new_im


# In[14]:


preprocessed_train_images = []
preprocessed_test_images = []
preprocessed_classified_images = []

for im in train_images:
    preprocessed_train_images.append(pad_and_resize(im))
for im in test_images:
    preprocessed_test_images.append(pad_and_resize(im))
for im in classified_train_images:
    preprocessed_classified_images.append(pad_and_resize(im))

# In[17]:
train_images[78]
preprocessed_train_images[78]

pickle.dump( preprocessed_train_images, open( "pkl/preprocessed_train_images.pkl", "wb" ) )
pickle.dump( preprocessed_test_images, open( "pkl/preprocessed_test_images.pkl", "wb" ) )
pickle.dump( preprocessed_classified_images, open( "pkl/preprocessed_classified_images.pkl", "wb" ) )

import numpy as np
import scipy.misc

import skimage.color
import skimage.exposure


def histBounds(histogram, margin=0.005):
    """
    Detect lower and upper boundary of histogram. This is used to stretch it afterwards to get better contrast.
    :param histogram: list containing the number of members for each index
    :param margin: count value lower than max(histogram)*margin will be considered as zero
    :return: (lowerMargin, upperMargin) indices in histogram array.
    """
    # cut off full black and full white pixels and ignore them
    lowCut = int(len(histogram) * 0.12)
    highCut = int(len(histogram) * 0.95)
    histCut = histogram[lowCut:highCut]
    logMargin = np.max(histCut) * margin
    #now search for minimum and maximum color
    nonzero = np.where(histCut > logMargin)
    # return the first and last index where the count is non zero
    return lowCut + np.min(nonzero), lowCut + np.max(nonzero)


def normalizeChannel(channel, maxVal=255.):
    """
    Normalizes the histogram of given 2D array (which is one cannel of a color image or grayscale).
    :param channel: the image data
    :param maxVal: maximum value which may occur in array
    """
    arr = channel.astype(float)
    hist, bounds = np.histogram(arr, maxVal)

    min, max = histBounds(hist)

    arr = (float(maxVal) / float(max - min)) * (arr - min)

    return np.clip(arr, 0, maxVal).astype(channel.dtype)


def normalizeLab(arr):
    """
    Normalizes (stretches) the histogram of given image data by the following steps:
    1. converting the image to Lab color system
    2. using the histogram of L channel to detect lower and upper bound
    3. stretch L channel to increase contrast
    4. convert data back to RGB image
    :param arr: image data returned from e.g. scipy.misc.fromimage
    :return: normalized image data
    """
    lab = skimage.color.rgb2lab(arr / 255.)

    lab[..., 0] = normalizeChannel(lab[..., 0], 100)

    rgb = skimage.color.lab2rgb(lab)
    rgb = (rgb * 255).astype(int)
    return rgb


def normalizeImage(img):
    """
    Normalizes the given PIL image
    :param img: an instance of PIL image
    :return: normalized PIL image
    """
    bytes = scipy.misc.fromimage(img)
    bytes = normalizeLab(bytes)
    return scipy.misc.toimage(bytes)


def normalizeImageChannel(img):
    """
    Normalizes the given PIL image
    :param img: an instance of PIL image
    :return: normalized PIL image
    """
    bytes = scipy.misc.fromimage(img)
    for i in range(1):
        bytes[..., i] = normalizeChannel(bytes[..., i], 255)
    return scipy.misc.toimage(bytes)

train_images[78]
normalizeImageChannel(train_images[78])
