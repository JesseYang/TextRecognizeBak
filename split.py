import os
import random

data_dir = 'generated_augment'
data_path_collections = [os.path.join(dirpath,filename)+'\n' for dirpath, dirnames, filenames in os.walk(data_dir) for filename in filenames if filename.endswith('png')]

dataset_name = 'generated_augment'
random.shuffle(data_path_collections)
# split into training set and test set
test_ratio = 0.1
total_num = len(data_path_collections)
test_num = int(test_ratio * total_num)
train_num = total_num - test_num
train_records = data_path_collections[0:train_num]
test_records = data_path_collections[train_num:]

# save to text file
all_out_file = open(dataset_name + '_all.txt', 'w')
for record in data_path_collections:
    all_out_file.write(record)
all_out_file.close()

train_out_file = open(dataset_name + '_train.txt', 'w')
print('train num ', len(train_records))
for record in train_records:
    train_out_file.write(record)
train_out_file.close()

test_out_file = open(dataset_name + '_test.txt', 'w')
print('test num ',len(test_records))
for record in test_records:
    test_out_file.write(record)
test_out_file.close()



small_test_file = open('1000_test.txt', 'w')
print('test num ',len(test_records[0:1000]))
for record in test_records[0:1000]:
    # print(record)
    small_test_file.write(record)