import os
import sys
import random
import shutil

root_dir = sys.argv[1]
out_dir = sys.argv[2]
if not os.path.exists(out_dir):
    os.mkdir(out_dir)
if not os.path.exists(os.path.join(out_dir, 'train')):
    os.mkdir(os.path.join(out_dir, 'train'))
if not os.path.exists(os.path.join(out_dir, 'valid')):
    os.mkdir(os.path.join(out_dir, 'valid'))
k = int(sys.argv[3])
label_list = os.listdir(root_dir)
for label in label_list:
    if not os.path.exists(os.path.join(out_dir, 'train', label)):
        os.mkdir(os.path.join(out_dir, 'train', label))
    if not os.path.exists(os.path.join(out_dir, 'valid', label)):
        os.mkdir(os.path.join(out_dir, 'valid', label))
    dir_path = os.path.join(root_dir, label)
    img_list = os.listdir(dir_path)
    img_num = len(img_list)
    random.shuffle(img_list)
    out_train_dir = os.path.join(out_dir, 'train', label)
    out_valid_dir = os.path.join(out_dir, 'valid', label)
    count = 0
    for img in img_list[:img_num / k]:
        count += 1
        img_path = os.path.join(dir_path, img)
        shutil.copy(img_path, out_valid_dir)
        print(count, "/", img_num)
    for img in img_list[img_num / k:]:
        count += 1
        img_path = os.path.join(dir_path, img)
        shutil.copy(img_path, out_train_dir)
        print(count, "/", img_num)
