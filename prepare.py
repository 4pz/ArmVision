import torch
import torchvision
import torchvision.transforms as transforms
from IPython.display import Image, clear_output  # to display images
from sklearn.model_selection import train_test_split
import yaml
from glob import glob
#from utils.google_utils import gdrive_download  # to download models/datasets

#print('Setup complete. Using torch %s %s' % (torch.__version__, torch.cuda.get_device_properties(0) if torch.cuda.is_available() else 'CPU'))

with open(r"dataset/data.yaml", 'r') as stream:
    num_classes = str(yaml.safe_load(stream)['nc'])

#print(num_classes, "number of classes")

image_list = glob(r"dataset/export/images/*.jpg")

#print(len(image_list), "image count")

train_img_list, val_img_list = train_test_split(image_list, test_size=0.2, random_state=2000)

#print(len(train_img_list))
#print(len(val_img_list))

with open(r'train.txt', 'w') as f:
    f.write('\n'.join(train_img_list) + '\n')

with open(r'val.txt', 'w') as f:
    f.write('\n'.join(val_img_list) + '\n')

with open(r'dataset/data.yaml', 'r') as f:
    data = yaml.load(f, Loader=yaml.FullLoader)

#print(data)

data['train'] = 'train.txt'
data['val'] = 'val.txt'

with open('dataset/data.yaml', 'w') as f:
    yaml.dump(data, f)

#print(data)
