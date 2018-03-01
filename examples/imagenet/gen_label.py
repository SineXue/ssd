import os
from random import shuffle


data_root = '/mnt/disk2/imagenet2015/ILSVRC2015/Data/CLS-LOC/'
val_anno_root = '/mnt/disk2/imagenet2015/ILSVRC2015/Annotations/CLS-LOC/val/'
train_root = data_root + 'train/'
val_data_root = data_root + 'val/'
cls_list = os.listdir(train_root)
name_cls_str = ''
name_cls_list = []
for i in range(len(cls_list)):
  cls = cls_list[i]
  cls_folder = train_root + cls
  img_list = os.listdir(cls_folder)
  for img_name in img_list:
    name_cls_list.append('%s/%s %d\n' % (cls, img_name, i))

shuffle(name_cls_list)
name_cls_list = name_cls_list[0:500000]
for name_cls in name_cls_list:
  name_cls_str += name_cls

with open('imgnet_train.txt', 'wb') as f:
  f.write(name_cls_str)

print 'Train file txt generated.'
name_cls_str = ''
name_cls_list = []
counter = 0
print 'totally %d images' % len(os.listdir(val_anno_root))
for val_anno_path in os.listdir(val_anno_root):
  counter += 1
  with open(val_anno_root + val_anno_path, 'rb') as f:
    val_anno = f.read()
  val_cls = cls_list.index(val_anno.split('name')[3][1:-2])
  val_data_path = val_anno_path.replace('xml', 'JPEG')
  name_cls_list.append('%s %d\n' % (val_data_path, val_cls))
  if not counter % 500:
    print counter

shuffle(name_cls_list)
for name_cls in name_cls_list:
  name_cls_str += name_cls

with open('imgnet_val.txt', 'wb') as f:
  f.write(name_cls_str)

print 'Val file txt generated'

# ./build/tools/convert_imageset /mnt/disk2/imagenet2015/ILSVRC2015/Data/CLS-LOC/train/
#  examples/imagenet/imgnet_train.txt