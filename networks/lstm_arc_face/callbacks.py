"""Helper for evaluation on the Labeled Faces in the Wild dataset
"""

# MIT License
#
# Copyright (c) 2016 David Sandberg
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import argparse
import sys
import sklearn
import cv2
import math
import datetime
import pickle
import mxnet as mx
import numpy as np

from mxnet import gluon
from mxnet import ndarray as nd

from scipy import misc
from scipy import interpolate

from sklearn.model_selection import KFold
from sklearn.decomposition import PCA

#################################################################################################################################
# Inspect that crap.
#################################################################################################################################
        def ver_test(nbatch):
          results = []
          for i in xrange(len(ver_list)):
            acc1, std1, acc2, std2, xnorm, embeddings_list = verification.test(ver_list[i], net, ctx, batch_size = args.batch_size)
            print('[%s][%d]XNorm: %f' % (ver_name_list[i], nbatch, xnorm))
            #print('[%s][%d]Accuracy: %1.5f+-%1.5f' % (ver_name_list[i], nbatch, acc1, std1))
            print('[%s][%d]Accuracy-Flip: %1.5f+-%1.5f' % (ver_name_list[i], nbatch, acc2, std2))
            results.append(acc2)
          return results

        def _batch_callback():
          mbatch = global_step[0]
          global_step[0]+=1

          if mbatch>0 and mbatch%args.verbose==0:
            save_step[0]+=1
            msave = save_step[0]
            do_save = False
            is_highest = False
            if args.task=='age' or args.task=='gender':
              acc = val_test(mbatch)
              if acc>=highest_acc[-1]:
                highest_acc[-1] = acc
                is_highest = True
                do_save = True
            else:
              acc_list = ver_test(mbatch)
              if len(acc_list)>0:
                lfw_score = acc_list[0]
                if lfw_score>highest_acc[0]:
                  highest_acc[0] = lfw_score
                  if lfw_score>=0.998:
                    do_save = True
                if acc_list[-1]>=highest_acc[-1]:
                  highest_acc[-1] = acc_list[-1]
                  if lfw_score>=0.99:
                    do_save = True
                    is_highest = True
            if args.ckpt==0:
              do_save = False
            elif args.ckpt>1:
              do_save = True
            if do_save:
              print('saving', msave)
              #print('saving gluon params')
              fname = os.path.join(args.prefix, 'model-gluon.params')
              net.save_params(fname)
              fname = os.path.join(args.prefix, 'model')
              net.export(fname, msave)
              #arg, aux = model.get_params()
              #mx.model.save_checkpoint(prefix, msave, model.symbol, arg, aux)
            print('[%d]Accuracy-Highest: %1.5f'%(mbatch, highest_acc[-1]))
          if args.max_steps>0 and mbatch>args.max_steps:
            sys.exit(0)


#################################################################################################################################
# This shit is verification
#################################################################################################################################


class LFold:
  def __init__(self, n_splits = 2, shuffle = False):
    self.n_splits = n_splits
    if self.n_splits>1:
      self.k_fold = KFold(n_splits = n_splits, shuffle = shuffle)

  def split(self, indices):
    if self.n_splits>1:
      return self.k_fold.split(indices)
    else:
      return [(indices, indices)]


def calculate_roc(thresholds, embeddings1, embeddings2, actual_issame, nrof_folds=10, pca = 0):
    assert(embeddings1.shape[0] == embeddings2.shape[0])
    assert(embeddings1.shape[1] == embeddings2.shape[1])
    nrof_pairs = min(len(actual_issame), embeddings1.shape[0])
    nrof_thresholds = len(thresholds)
    k_fold = LFold(n_splits=nrof_folds, shuffle=False)

    tprs = np.zeros((nrof_folds,nrof_thresholds))
    fprs = np.zeros((nrof_folds,nrof_thresholds))
    accuracy = np.zeros((nrof_folds))
    indices = np.arange(nrof_pairs)
    #print('pca', pca)

    if pca==0:
      diff = np.subtract(embeddings1, embeddings2)
      dist = np.sum(np.square(diff),1)

    for fold_idx, (train_set, test_set) in enumerate(k_fold.split(indices)):
        #print('train_set', train_set)
        #print('test_set', test_set)
        if pca>0:
          print('doing pca on', fold_idx)
          embed1_train = embeddings1[train_set]
          embed2_train = embeddings2[train_set]
          _embed_train = np.concatenate( (embed1_train, embed2_train), axis=0 )
          #print(_embed_train.shape)
          pca_model = PCA(n_components=pca)
          pca_model.fit(_embed_train)
          embed1 = pca_model.transform(embeddings1)
          embed2 = pca_model.transform(embeddings2)
          embed1 = sklearn.preprocessing.normalize(embed1)
          embed2 = sklearn.preprocessing.normalize(embed2)
          #print(embed1.shape, embed2.shape)
          diff = np.subtract(embed1, embed2)
          dist = np.sum(np.square(diff),1)

        # Find the best threshold for the fold
        acc_train = np.zeros((nrof_thresholds))
        for threshold_idx, threshold in enumerate(thresholds):
            _, _, acc_train[threshold_idx] = calculate_accuracy(threshold, dist[train_set], actual_issame[train_set])
        best_threshold_index = np.argmax(acc_train)
        #print('threshold', thresholds[best_threshold_index])
        for threshold_idx, threshold in enumerate(thresholds):
            tprs[fold_idx,threshold_idx], fprs[fold_idx,threshold_idx], _ = calculate_accuracy(threshold, dist[test_set], actual_issame[test_set])
        _, _, accuracy[fold_idx] = calculate_accuracy(thresholds[best_threshold_index], dist[test_set], actual_issame[test_set])

    tpr = np.mean(tprs,0)
    fpr = np.mean(fprs,0)
    return tpr, fpr, accuracy

def calculate_accuracy(threshold, dist, actual_issame):
    predict_issame = np.less(dist, threshold)
    tp = np.sum(np.logical_and(predict_issame, actual_issame))
    fp = np.sum(np.logical_and(predict_issame, np.logical_not(actual_issame)))
    tn = np.sum(np.logical_and(np.logical_not(predict_issame), np.logical_not(actual_issame)))
    fn = np.sum(np.logical_and(np.logical_not(predict_issame), actual_issame))

    tpr = 0 if (tp+fn==0) else float(tp) / float(tp+fn)
    fpr = 0 if (fp+tn==0) else float(fp) / float(fp+tn)
    acc = float(tp+tn)/dist.size
    return tpr, fpr, acc



def calculate_val(thresholds, embeddings1, embeddings2, actual_issame, far_target, nrof_folds=10):
    assert(embeddings1.shape[0] == embeddings2.shape[0])
    assert(embeddings1.shape[1] == embeddings2.shape[1])
    nrof_pairs = min(len(actual_issame), embeddings1.shape[0])
    nrof_thresholds = len(thresholds)
    k_fold = LFold(n_splits=nrof_folds, shuffle=False)

    val = np.zeros(nrof_folds)
    far = np.zeros(nrof_folds)

    diff = np.subtract(embeddings1, embeddings2)
    dist = np.sum(np.square(diff),1)
    indices = np.arange(nrof_pairs)

    for fold_idx, (train_set, test_set) in enumerate(k_fold.split(indices)):

        # Find the threshold that gives FAR = far_target
        far_train = np.zeros(nrof_thresholds)
        for threshold_idx, threshold in enumerate(thresholds):
            _, far_train[threshold_idx] = calculate_val_far(threshold, dist[train_set], actual_issame[train_set])
        if np.max(far_train)>=far_target:
            f = interpolate.interp1d(far_train, thresholds, kind='slinear')
            threshold = f(far_target)
        else:
            threshold = 0.0

        val[fold_idx], far[fold_idx] = calculate_val_far(threshold, dist[test_set], actual_issame[test_set])

    val_mean = np.mean(val)
    far_mean = np.mean(far)
    val_std = np.std(val)
    return val_mean, val_std, far_mean


def calculate_val_far(threshold, dist, actual_issame):
    predict_issame = np.less(dist, threshold)
    true_accept = np.sum(np.logical_and(predict_issame, actual_issame))
    false_accept = np.sum(np.logical_and(predict_issame, np.logical_not(actual_issame)))
    n_same = np.sum(actual_issame)
    n_diff = np.sum(np.logical_not(actual_issame))
    #print(true_accept, false_accept)
    #print(n_same, n_diff)
    val = float(true_accept) / float(n_same)
    far = float(false_accept) / float(n_diff)
    return val, far

def evaluate(embeddings, actual_issame, nrof_folds=10, pca = 0):
    # Calculate evaluation metrics
    thresholds = np.arange(0, 4, 0.01)
    embeddings1 = embeddings[0::2]
    embeddings2 = embeddings[1::2]
    tpr, fpr, accuracy = calculate_roc(thresholds, embeddings1, embeddings2,
        np.asarray(actual_issame), nrof_folds=nrof_folds, pca = pca)
    thresholds = np.arange(0, 4, 0.001)
    val, val_std, far = calculate_val(thresholds, embeddings1, embeddings2,
        np.asarray(actual_issame), 1e-3, nrof_folds=nrof_folds)
    return tpr, fpr, accuracy, val, val_std, far

def load_bin(path, image_size):
  bins, issame_list = pickle.load(open(path, 'rb'))
  data_list = []
  for flip in [0,1]:
    data = nd.empty((len(issame_list)*2, 3, image_size[0], image_size[1]))
    data_list.append(data)
  for i in xrange(len(issame_list)*2):
    _bin = bins[i]
    img = mx.image.imdecode(_bin)
    img = nd.transpose(img, axes=(2, 0, 1))
    for flip in [0,1]:
      if flip==1:
        img = mx.ndarray.flip(data=img, axis=2)
      data_list[flip][i][:] = img
    if i%1000==0:
      print('loading bin', i)
  print(data_list[0].shape)
  return (data_list, issame_list)

def test(data_set, net, ctx, batch_size, nfolds=10):
  print('testing verification..')
  data_list = data_set[0]
  issame_list = data_set[1]
  embeddings_list = []
  time_consumed = 0.0
  for i in xrange( len(data_list) ):
    data = data_list[i]
    embeddings = None
    ba = 0
    while ba<data.shape[0]:
      bb = min(ba+batch_size, data.shape[0])
      count = bb-ba
      #print(ba, bb)
      x = nd.slice_axis(data, axis=0, begin=bb-batch_size, end=bb)
      #print(_data.shape, _label.shape)
      time0 = datetime.datetime.now()
      #x = x.as_in_context(ctx[0])
      xs = gluon.utils.split_and_load(x, ctx_list=ctx, batch_axis=0)
      zs = []
      for x in xs:
        with mx.autograd.predict_mode():
          z = net.feature(x)
        zs.append(z)
      zss = []
      for z in zs:
        zss.append(z.asnumpy())
      zss = np.concatenate(zss, axis=0)
      #print(zss.shape)
      _embeddings = zss
      #_arg, _aux = model.get_params()
      #__arg = {}
      #for k,v in _arg.iteritems():
      #  __arg[k] = v.as_in_context(_ctx)
      #_arg = __arg
      #_arg["data"] = _data.as_in_context(_ctx)
      #_arg["softmax_label"] = _label.as_in_context(_ctx)
      #for k,v in _arg.iteritems():
      #  print(k,v.context)
      #exe = sym.bind(_ctx, _arg ,args_grad=None, grad_req="null", aux_states=_aux)
      #exe.forward(is_train=False)
      #net_out = exe.outputs
      #_embeddings = z.asnumpy()
      time_now = datetime.datetime.now()
      diff = time_now - time0
      time_consumed+=diff.total_seconds()
      #print(_embeddings.shape)
      if embeddings is None:
        embeddings = np.zeros( (data.shape[0], _embeddings.shape[1]) )
      embeddings[ba:bb,:] = _embeddings[(batch_size-count):,:]
      ba = bb
    embeddings_list.append(embeddings)

  _xnorm = 0.0
  _xnorm_cnt = 0
  for embed in embeddings_list:
    for i in xrange(embed.shape[0]):
      _em = embed[i]
      _norm=np.linalg.norm(_em)
      #print(_em.shape, _norm)
      _xnorm+=_norm
      _xnorm_cnt+=1
  _xnorm /= _xnorm_cnt

  embeddings = embeddings_list[0].copy()
  embeddings = sklearn.preprocessing.normalize(embeddings)
  acc1 = 0.0
  std1 = 0.0
  #_, _, accuracy, val, val_std, far = evaluate(embeddings, issame_list, nrof_folds=10)
  #acc1, std1 = np.mean(accuracy), np.std(accuracy)

  #print('Validation rate: %2.5f+-%2.5f @ FAR=%2.5f' % (val, val_std, far))
  #embeddings = np.concatenate(embeddings_list, axis=1)
  embeddings = embeddings_list[0] + embeddings_list[1]
  embeddings = sklearn.preprocessing.normalize(embeddings)
  print(embeddings.shape)
  print('infer time', time_consumed)
  _, _, accuracy, val, val_std, far = evaluate(embeddings, issame_list, nrof_folds=nfolds)
  acc2, std2 = np.mean(accuracy), np.std(accuracy)
  return acc1, std1, acc2, std2, _xnorm, embeddings_list
