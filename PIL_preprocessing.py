#!/usr/bin/env python
# coding: utf-8

# In[11]:


from PIL import Image, ImageOps, ImageFilter
import pickle
import numpy as np

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
    new_im = new_im.convert('L')
    # new_im = im.resize((desired_size, desired_size), Image.ANTIALIAS)
    return new_im

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

pickle.dump( preprocessed_train_images, open( "pkl/train_padded64", "wb" ) )
pickle.dump( preprocessed_test_images, open( "pkl/test_padded64.pkl", "wb" ) )
pickle.dump( preprocessed_classified_images, open( "pkl/classified_padded64.pkl", "wb" ) )

def pad_and_resize(im):
    desired_size = 64
    # old_size = im.size  # old_size[0] is in (width, height) format
    # ratio = float(desired_size)/max(old_size)
    # new_size = tuple([int(x*ratio) for x in old_size])
    # im = im.resize(new_size, Image.ANTIALIAS)
    # new_im = Image.new("RGB", (desired_size, desired_size), "white")
    # new_im.paste(im, ((desired_size-new_size[0])//2,
    #                     (desired_size-new_size[1])//2))
    # new_im = new_im.convert('L')
    new_im = im.resize((desired_size, desired_size), Image.ANTIALIAS)
    return new_im

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

pickle.dump( preprocessed_train_images, open( "pkl/train_resized64", "wb" ) )
pickle.dump( preprocessed_test_images, open( "pkl/test_resized64.pkl", "wb" ) )
pickle.dump( preprocessed_classified_images, open( "pkl/classified_resized64.pkl", "wb" ) )





######======80========######
def pad_and_resize(im):
    desired_size = 80
    old_size = im.size  # old_size[0] is in (width, height) format
    ratio = float(desired_size)/max(old_size)
    new_size = tuple([int(x*ratio) for x in old_size])
    im = im.resize(new_size, Image.ANTIALIAS)
    new_im = Image.new("RGB", (desired_size, desired_size), "white")
    new_im.paste(im, ((desired_size-new_size[0])//2,
                        (desired_size-new_size[1])//2))
    new_im = new_im.convert('L')
    # new_im = im.resize((desired_size, desired_size), Image.ANTIALIAS)
    return new_im

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

pickle.dump( preprocessed_train_images, open( "pkl/train_padded80", "wb" ) )
pickle.dump( preprocessed_test_images, open( "pkl/test_padded80.pkl", "wb" ) )
pickle.dump( preprocessed_classified_images, open( "pkl/classified_padded80.pkl", "wb" ) )

def pad_and_resize(im):
    desired_size = 80
    # old_size = im.size  # old_size[0] is in (width, height) format
    # ratio = float(desired_size)/max(old_size)
    # new_size = tuple([int(x*ratio) for x in old_size])
    # im = im.resize(new_size, Image.ANTIALIAS)
    # new_im = Image.new("RGB", (desired_size, desired_size), "white")
    # new_im.paste(im, ((desired_size-new_size[0])//2,
    #                     (desired_size-new_size[1])//2))
    # new_im = new_im.convert('L')
    new_im = im.resize((desired_size, desired_size), Image.ANTIALIAS)
    return new_im

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

pickle.dump( preprocessed_train_images, open( "pkl/train_resized80", "wb" ) )
pickle.dump( preprocessed_test_images, open( "pkl/test_resized80.pkl", "wb" ) )
pickle.dump( preprocessed_classified_images, open( "pkl/classified_resized80.pkl", "wb" ) )
