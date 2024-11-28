from typing import Any
import numpy as np
import os
from utils.utils import NdpKernel, make_memory_map, pad8, pad16
import configs

batch_size = 1
cols = 4
rows = 4

n_effective_classes = 16
n_features = 16

# Multinomial Naive Bayes
# 
# update_feature_log_prob(self, alpha):
# alpha = 1.0
# alpha : float amount of smoothing to apply (0. means no smoothing)
#
# smoothed_fc = self.feature_count_ + alpha
# smoothed_cc = smoothed_fc.sum(axis=1).reshape(-1, 1)
# self.feature_log_prob_ = cp.log(smoothed_fc) - cp.log(
#     smoothed_cc.reshape(-1, 1)
#
#
# joint_log_likelihood(self, X):
# X : array-like of size (n_samples, n_features)
# 
# X = self._check_X(X)
# jll = self._joint_log_likelihood(X)
# indices = cp.argmax(jll, axis=1)
# y_hat = invert_labels(indices, self.classes_)
# y_hat = CumlArray(data=y_hat, index=index)
# return y_hat
#
# joint_log_likelihood(self, X):
# X : array-like of size (n_samples, n_features)
#
# X's shape : (batch_size, cols, rows)
# self.feature_log_prob_ : array-like of size (n_classes, n_features)
# self.class_log_prior_ : array-like of size (n_classes,)
#
# ret = X.dot(self.feature_log_prob_.T)
# ret += self.class_log_prior_
# return ret


class NaiveBayesKernel0(NdpKernel):
  def __init__(self):
    super().__init__()
    self.vector_size = cols * rows * batch_size
    self.node_num = cols * rows
    self.batch_size = batch_size
    self.input_x_addr = 0x100000000000000
    self.feature_log_prob_addr = 0x101000000000000
    self.class_log_prior_addr = 0x102000000000000
    self.sync = 0
    self.kernel_id = 0
    self.kernel_name = f'naivebayes_kernel0'
    self.base_addr = self.input_x_addr
    self.bound = self.vector_size * configs.data_size
    self.input_addrs = [self.input_x_addr, self.node_num, self.batch_size]

    # X
    x = np.random.rand(batch_size, n_effective_classes, n_features).astype(np.float16)
    print("x : ")
    print(x)

    # feature_log_prob_
    alpha = 1.0
    feature_cnt = np.random.rand(n_effective_classes, n_features).astype(np.float16)
    smoothed_fc = feature_cnt + alpha
    smoothed_cc = smoothed_fc.sum(axis=1).reshape(-1, 1)
    feature_log_prob_ = np.log(smoothed_fc) - np.log(smoothed_cc.reshape(-1, 1))
    print("feature_log_prob_ : ")
    print(feature_log_prob_)

    # class_log_prior_
    class_count = np.random.rand(n_effective_classes).astype(np.float16)
    log_class_count = np.log(class_count)
    class_log_prior_ = log_class_count - np.log(class_count.sum())

    class_log_prior_ = class_log_prior_.reshape(1, -1)
    print("class_log_prior : ")
    print(class_log_prior_)

    # joint log likelihood(x)
    ret = np.dot(x, feature_log_prob_.T)
    print("x.dot(feature_log_prob_.T) : ")
    print(ret)

    ret += class_log_prior_
    print("ret : ")
    print(ret)

    indices = np.argmax(ret, axis=1)
    print("indices : ")
    print(indices)

    self.input_data = x
    self.feature_log_prob = pad16(np.array(feature_log_prob_.reshape(n_effective_classes * n_features), dtype=np.float16))
    self.class_log_prior = pad16(np.array(class_log_prior_.reshape(n_effective_classes), dtype=np.float16))
        
    
  def make_kernel(self):
    template = ''
    template += f'-kernel name = {self.kernel_name}\n'
    template += f'-kernel id = {self.kernel_id}\n'
    template += '\n'
    
    template += f'vsetvli t0, a0, e32, m2\n'
    template += f'li x1, {configs.spad_addr}\n'

    template += f'KERNELBODY:\n'
    template += f'vsetvli t0, a0, e16, m1\n'
    template += f'vle16.v v1, (ADDR)\n'
    
    return template
  
  def make_input_map(self):
   return make_memory_map([
        (self.input_x_addr, self.input_data),
        (self.feature_log_prob_addr, self.feature_log_prob),
        (self.class_log_prior_addr, self.class_log_prior),
    ])
  
  def make_output_map(self):
    return make_memory_map([
        # (self.input_info_addr, self.input_info),
        # (self.input_x_addr, self.input_data),
      ])
  
