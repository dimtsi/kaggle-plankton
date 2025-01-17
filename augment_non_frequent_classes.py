import pickle
import numpy as np
import pandas as pd
from torchvision import transforms
from PIL import Image, ImageOps, ImageFilter
import matplotlib.pyplot as plt

"""
Create augmented external dataset with intense augmentation
on non-frequent classes
"""

original_train_images = pickle.load(open("pkl/classified_train_images.pkl", "rb"))
original_train_labels = pickle.load(open("pkl/classified_train_labels.pkl", "rb"))
original_test_images = pickle.load(open("pkl/test_padded64.pkl", "rb"))
original_test_labels = pickle.load(open("pkl/test_filenames.pkl", "rb"))

##create separate test set
test_set_mine_indexes = pickle.load(open("pkl/test_set_mine_indexes_classified.pkl", "rb"))
train_images = [i for j, i in enumerate(original_train_images) if j not in test_set_mine_indexes]
train_labels = [i for j, i in enumerate(original_train_labels) if j not in test_set_mine_indexes]

test_mine_images = [i for j, i in enumerate(original_train_images) if j in test_set_mine_indexes]
test_mine_labels = [i for j, i in enumerate(original_train_labels) if j in test_set_mine_indexes]


len(original_train_images)

class_weights = {}
class_sample_counts = np.bincount(train_labels)
train_samples_weight = [class_sample_counts[class_id] for class_id in train_labels]


most_frequent = np.max(class_sample_counts)
most_frequent
augmentation_rate_per_class = most_frequent//class_sample_counts//7
augmentation_rate_per_class


transform = transforms.Compose([
    transforms.RandomRotation(degrees=360),
    transforms.RandomAffine(50, shear=20),


augmented_images = []
augmented_labels = []
for i, image in enumerate(train_images):
    label = train_labels[i]
    for num_of_transforms in range(augmentation_rate_per_class[label]):
        augmented_images.append(transform(image))
        augmented_labels.append(label)

pickle.dump(augmented_images, open("pkl/augmented/augmented_only_images.pkl", "wb" ))
pickle.dump(augmented_labels, open("pkl/augmented/augmented_only_labels.pkl", "wb" ))


all_images_with_aug = train_images+augmented_images
len(all_images_with_aug)
all_labels = train_labels+augmented_labels


pickle.dump(all_images_with_aug, open("pkl/augmented/all_images.pkl", "wb" ))
pickle.dump(all_labels, open("pkl/augmented/all_labels.pkl", "wb" ))



classified_train_images = []
classified_train_labels = []

for x in range(len(all_images_with_aug)):
    if all_labels[x] == 0:
        continue
    else:
        classified_train_images.append(all_images_with_aug[x])
        classified_train_labels.append(all_labels[x])

pickle.dump( classified_train_images, open("pkl/augmented/classified_all_images.pkl", "wb" ))
pickle.dump( classified_train_labels, open("pkl/augmented/classified_all_labels.pkl", "wb"))



train_images = pickle.load(open("pkl/augmented/all_images.pkl", "rb"))
classified_train_images = pickle.load(open("pkl/augmented/classified_all_images.pkl", "rb"))


def pad_and_resize(im):
    desired_size = 64
    new_im = im.resize((desired_size, desired_size), Image.ANTIALIAS)
    return new_im

preprocessed_train_images = []
preprocessed_test_images = []
preprocessed_classified_images = []

for im in train_images:
    preprocessed_train_images.append(pad_and_resize(im))
for im in classified_train_images:
    preprocessed_classified_images.append(pad_and_resize(im))

train_images[78]
preprocessed_train_images[78]

pickle.dump( preprocessed_train_images, open( "pkl/augmented/train_resized64", "wb" ) )
pickle.dump( preprocessed_classified_images, open( "pkl/augmented/classified_resized64.pkl", "wb" ) )



train_images = pickle.load(open("pkl/augmented/all_images.pkl", "rb"))
classified_train_images = pickle.load(open("pkl/augmented/classified_all_images.pkl", "rb"))

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
    return new_im

preprocessed_train_images = []
preprocessed_classified_images = []

for im in train_images:
    preprocessed_train_images.append(pad_and_resize(im))
for im in classified_train_images:
    preprocessed_classified_images.append(pad_and_resize(im))

train_images[78]

pickle.dump( preprocessed_train_images, open( "pkl/augmented/train_padded64", "wb" ) )
pickle.dump( preprocessed_classified_images, open( "pkl/augmented/classified_padded64.pkl", "wb" ) )

def pad_and_resize(im):
    desired_size = 64
    new_im = im.resize((desired_size, desired_size), Image.ANTIALIAS)
    return new_im

preprocessed_train_images = []
preprocessed_classified_images = []

for im in train_images:
    preprocessed_train_images.append(pad_and_resize(im))
for im in classified_train_images:
    preprocessed_classified_images.append(pad_and_resize(im))

# In[17]:
train_images[78]
preprocessed_train_images[78]

pickle.dump( preprocessed_train_images, open( "pkl/augmented/train_resized64", "wb" ) )
pickle.dump( preprocessed_classified_images, open( "pkl/augmented/classified_resized64.pkl", "wb" ) )


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
    return new_im

preprocessed_train_images = []
preprocessed_classified_images = []

for im in train_images:
    preprocessed_train_images.append(pad_and_resize(im))
for im in classified_train_images:
    preprocessed_classified_images.append(pad_and_resize(im))


pickle.dump( preprocessed_train_images, open( "pkl/augmented/train_padded80", "wb" ) )
pickle.dump( preprocessed_classified_images, open( "pkl/augmented/classified_padded80.pkl", "wb" ) )

def pad_and_resize(im):
    desired_size = 80
    new_im = im.resize((desired_size, desired_size), Image.ANTIALIAS)
    return new_im

preprocessed_train_images = []
preprocessed_classified_images = []

for im in train_images:
    preprocessed_train_images.append(pad_and_resize(im))
for im in classified_train_images:
    preprocessed_classified_images.append(pad_and_resize(im))

train_images[78]
preprocessed_train_images[78]

pickle.dump( preprocessed_train_images, open( "pkl/augmented/train_resized80", "wb" ) )
pickle.dump( preprocessed_classified_images, open( "pkl/augmented/classified_resized80.pkl", "wb" ) )
