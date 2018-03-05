import os

from scipy import misc
import json
import sys
import shutil
import numpy as np
import numpy.ma as ma
import random
# print(sys.path)
from MessUp.MessUp.operations import *
from augmentor import Augmentor
# try:
#     from MessUp.MessUp.operations import *
# except Exception:
#     from .MessUp.MessUp.operations import *

raw_data_dir = 'generated/hard_sammple_minning'
target_data_dir = 'generated_augment/hard_sammple_minning'
# initalize augmentors
dislocate = Dislocate(r_scale = (30, 100))
resize = Resize(dsize = (30, None))
affine = Affine(r_degree = (-20, 20), direction = 'x')

# clear
if os.path.exists(target_data_dir):
    shutil.rmtree(target_data_dir)
os.mkdir(target_data_dir)

# collect raw data
files = os.listdir(raw_data_dir)
aug = Augmentor()

for idx, file in enumerate(files):
    # if idx > 1:
    #     continue
    print("{}/{}".format(idx+1, len(files)))
    if file.endswith('txt'):
        shutil.copy(os.path.join(raw_data_dir, file), os.path.join(target_data_dir, file))
        continue

    raw_data_path = os.path.join(raw_data_dir, file)
    img = misc.imread(raw_data_path)

    # perform augmentors
    img = resize(img)
    r = round(random.uniform(0, 1), 1)
    if r < 0.3:
        img = dislocate(img)
    r = round(random.uniform(0, 1), 1)
    if r < 0.3:
        img = affine(img)
    target_path = os.path.join(target_data_dir, file)
    print(target_path)


    if random.uniform(0, 1) >= 0.9:
        img = aug.vignetting(img)
    else:
        img = aug.gaussian_blur(img)



    misc.imsave(target_path, img)


    # break

    # target_data_path_collections.append(target_path+'\n')
    # print(idx, num_raw_data, sep='/')

'''
# split into training set and test set
test_ratio = 0.1
total_num = len(target_data_path_collections)
test_num = int(test_ratio * total_num)
train_num = total_num - test_num
train_records = target_data_path_collections[0:train_num]
test_records = target_data_path_collections[train_num:]

# save to text file
all_out_file = open(dataset_name + '_all.txt', 'w')
for record in target_data_path_collections:
    all_out_file.write(record)
all_out_file.close()

train_out_file = open(dataset_name + '_train.txt', 'w')
for record in train_records:
    train_out_file.write(record)
train_out_file.close()

test_out_file = open(dataset_name + '_test.txt', 'w')
for record in test_records:
    test_out_file.write(record)
test_out_file.close()
'''
