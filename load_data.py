import pickle
from PIL import Image

#labels with the same order
train_images = []
train_labels = []

test_images = []
test_dict = {}
train_filenames = []
test_filenames = []


labels_df = pd.read_csv('train_onelabel.csv')
labels_dict = labels_df.set_index('image')['class'].to_dict()

for filename in labels_df['image'].values: ##to keep mapping with classes
    train_images.append(Image.open('train_images/'+filename).copy())
    train_labels.append(labels_dict[filename])
    train_filenames.append(filename)
for filename in glob.iglob('test_images' +'/*'):
    image = Image.open(filename).copy()
    test_images.append(image)
    test_filenames.append(filename.replace('test_images/', ''))
    
pickle.dump( train_images, open( "pkl/train_images.pkl", "wb" ) )
pickle.dump( train_labels, open( "pkl/train_labels.pkl", "wb" ) )
pickle.dump( train_filenames, open( "pkl/train_filenames.pkl", "wb" ) )
pickle.dump( test_images, open( "pkl/test_images.pkl", "wb" ) )
pickle.dump( test_filenames, open( "pkl/test_filenames.pkl", "wb" ) )