# -*- coding: utf-8 -*-
from __future__ import print_function
import caffe
from caffe.model_libs import *
from google.protobuf import text_format

import math
import os
import shutil
import stat
import subprocess
import sys


# Add extra layers on top of a "base" network (e.g. VGGNet or Inception).
def AddExtraLayers(net, use_batchnorm=True):

        # Add additional layers to handle atrous effect.
        last_layer = net.keys()[-1]

        # 10 x 10
        Res18Body(net, last_layer, '6', out2a=128, out2b=128, stride=2, kernel=2, use_branch1=True)
        # 5 x 5
        Res18Body(net, 'res6', '7', out2a=128, out2b=128, stride=2, kernel=2, use_branch1=True)
        # 3 x 3
        Res18Body(net, 'res7', '8', out2a=64, out2b=64, stride=2, kernel=2, use_branch1=True)
        # 2 x 2
        Res18Body(net, 'res8', '9', out2a=64, out2b=64, stride=2, kernel=2, use_branch1=True)
        # 1 x 1
        return net


caffe_root = os.getcwd()

# Set true if you want to start training right after generating all files.
# run_soon = True
# Set true if you want to load from most recently saved snapshot.
# Otherwise, we will load from the pretrain_model defined below.
# resume_training = True
# If true, Remove old model files.
# remove_old_models = False
train_data = "examples/VOC0712/VOC0712_trainval_lmdb"
resize_width = 300
resize_height = 300
resize = "{}x{}".format(resize_width, resize_height)
batch_sampler = [
        {
                'sampler': {
                        },
                'max_trials': 1,
                'max_sample': 1,
        },
        {
                'sampler': {
                        'min_scale': 0.3,
                        'max_scale': 1.0,
                        'min_aspect_ratio': 0.5,
                        'max_aspect_ratio': 2.0,
                        },
                'sample_constraint': {
                        'min_jaccard_overlap': 0.1,
                        },
                'max_trials': 50,
                'max_sample': 1,
        },
        {
                'sampler': {
                        'min_scale': 0.3,
                        'max_scale': 1.0,
                        'min_aspect_ratio': 0.5,
                        'max_aspect_ratio': 2.0,
                        },
                'sample_constraint': {
                        'min_jaccard_overlap': 0.3,
                        },
                'max_trials': 50,
                'max_sample': 1,
        },
        {
                'sampler': {
                        'min_scale': 0.3,
                        'max_scale': 1.0,
                        'min_aspect_ratio': 0.5,
                        'max_aspect_ratio': 2.0,
                        },
                'sample_constraint': {
                        'min_jaccard_overlap': 0.5,
                        },
                'max_trials': 50,
                'max_sample': 1,
        },
        {
                'sampler': {
                        'min_scale': 0.3,
                        'max_scale': 1.0,
                        'min_aspect_ratio': 0.5,
                        'max_aspect_ratio': 2.0,
                        },
                'sample_constraint': {
                        'min_jaccard_overlap': 0.7,
                        },
                'max_trials': 50,
                'max_sample': 1,
        },
        {
                'sampler': {
                        'min_scale': 0.3,
                        'max_scale': 1.0,
                        'min_aspect_ratio': 0.5,
                        'max_aspect_ratio': 2.0,
                        },
                'sample_constraint': {
                        'min_jaccard_overlap': 0.9,
                        },
                'max_trials': 50,
                'max_sample': 1,
        },
        {
                'sampler': {
                        'min_scale': 0.3,
                        'max_scale': 1.0,
                        'min_aspect_ratio': 0.5,
                        'max_aspect_ratio': 2.0,
                        },
                'sample_constraint': {
                        'max_jaccard_overlap': 1.0,
                        },
                'max_trials': 50,
                'max_sample': 1,
        },
        ]
train_transform_param = {
        'mirror': True,
        'crop_size': 300,
        'mean_value': [104, 117, 123],
        'resize_param': {
                'prob': 1,
                'resize_mode': P.Resize.WARP,
                'height': resize_height,
                'width': resize_width,
                'interp_mode': [
                        P.Resize.LINEAR,
                        P.Resize.AREA,
                        P.Resize.NEAREST,
                        P.Resize.CUBIC,
                        P.Resize.LANCZOS4,
                        ],
                },
        'distort_param': {
                'brightness_prob': 0.5,
                'brightness_delta': 32,
                'contrast_prob': 0.5,
                'contrast_lower': 0.5,
                'contrast_upper': 1.5,
                'hue_prob': 0.5,
                'hue_delta': 18,
                'saturation_prob': 0.5,
                'saturation_lower': 0.5,
                'saturation_upper': 1.5,
                'random_order_prob': 0.0,
                },
        'expand_param': {
                'prob': 0.5,
                'max_expand_ratio': 4.0,
                },
        'emit_constraint': {
            'emit_type': caffe_pb2.EmitConstraint.CENTER,
            }
        }
job_name = "SSD_{}".format(resize)
save_dir = "models/ResNet/ResNet-18/{}".format(job_name)
train_net_file = "{}/train.prototxt".format(save_dir)

label_map_file = "data/VOC0712/labelmap_voc.prototxt"

# MultiBoxLoss parameters.
num_classes = 21
share_location = True
background_label_id=0
train_on_diff_gt = True
normalization_mode = P.Loss.VALID
code_type = P.PriorBox.CENTER_SIZE
neg_pos_ratio = 3.
loc_weight = (neg_pos_ratio + 1.) / 4.
mining_type = P.MultiBoxLoss.MAX_NEGATIVE
multibox_loss_param = {
    'loc_loss_type': P.MultiBoxLoss.SMOOTH_L1,
    'conf_loss_type': P.MultiBoxLoss.SOFTMAX,
    'loc_weight': loc_weight,
    'num_classes': num_classes,
    'share_location': share_location,
    'match_type': P.MultiBoxLoss.PER_PREDICTION,
    'overlap_threshold': 0.5,
    'use_prior_for_matching': True,
    'background_label_id': background_label_id,
    'use_difficult_gt': train_on_diff_gt,
    'mining_type': mining_type,
    'neg_pos_ratio': neg_pos_ratio,
    'neg_overlap': 0.5,
    'code_type': code_type,
    }
loss_param = {
    'normalization': normalization_mode,
    }

# parameters for generating priors.
# minimum dimension of input image
min_dim = 300
mbox_source_layers = ['res4b_relu', 'res5b_relu', 'res6_relu', 'res7_relu','res8_relu', 'res9_relu']
# in percent %
min_ratio = 20
max_ratio = 95
step = int(math.floor((max_ratio - min_ratio) / (len(mbox_source_layers) - 1)))
min_sizes = []
max_sizes = []
for ratio in xrange(min_ratio + step, max_ratio + 1, step):
  min_sizes.append(min_dim * ratio / 100.)
  temp = min_dim * (ratio + step) / 100.
  if temp > 300:
    temp = 300
  max_sizes.append(temp)
min_sizes = [min_dim * 20 / 100.] + min_sizes
max_sizes = [[]] + max_sizes
aspect_ratios = [[2], [2, 3], [2, 3], [2, 3], [2, 3], [2, 3]]
steps = [8, 16, 32, 64, 100, 300]
# variance used to encode/decode prior bboxes.
if code_type == P.PriorBox.CENTER_SIZE:
  prior_variance = [0.1, 0.1, 0.2, 0.2]
else:
  prior_variance = [0.1]
flip = True
clip = False


def train_net():
  n = caffe.NetSpec()
  n.data, n.label = CreateAnnotatedDataLayer(train_data, batch_size=32,
                                                 train=True, output_label=True, label_map_file=label_map_file,
                                                 transform_param=train_transform_param, batch_sampler=batch_sampler)
  ResNet18Body(n, from_layer='data', use_pool5=False, use_dilation_conv5=False)
  AddExtraLayers(n, use_batchnorm=True)
  # Don't use batch norm for location/confidence prediction layers.
  mbox_layers = CreateMultiBoxHead(n, data_layer='data', from_layers=mbox_source_layers,
                                   use_batchnorm=False, min_sizes=min_sizes, max_sizes=max_sizes,
                                   aspect_ratios=aspect_ratios, steps=steps, num_classes=num_classes, share_location=share_location,
                                   flip=flip, clip=clip, prior_variance=prior_variance, kernel_size=3, pad=1)
  # Create the MultiBoxLossLayer.
  name = "mbox_loss"
  mbox_layers.append(n.label)
  n[name] = L.MultiBoxLoss(*mbox_layers, multibox_loss_param=multibox_loss_param,
                             loss_param=loss_param, include=dict(phase=caffe_pb2.Phase.Value('TRAIN')),
                             propagate_down=[True, True, False, False])
  return n.to_proto()

with open(train_net_file, 'w') as f:
  f.write('name: "ResNet-18-train"\n')
  f.write(str(train_net()))
