#!/usr/bin/python
# gpu_hello.py
  
import argparse
import os

########################
# set env variables using ``os.environ``;do the following before initializing TensorFlow to limit TensorFlow to first GPU
########################
#os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
#os.environ["CUDA_VISIBLE_DEVICES"]="0"

##############################
# you can double check that you have the correct devices visible to TF
##############################
import tensorflow as tf
from tensorflow.python.client import device_lib
print("-------------------> Now we print device_lib.list_local_devices()")
print(device_lib.list_local_devices())
print('')

##############################
# log device placements
##############################
# Creates a graph.
a = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], shape=[2, 3], name='a')
b = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], shape=[3, 2], name='b')
c = tf.matmul(a, b)
# Creates a session with log_device_placement set to True.
sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))
# Runs the op.
print('--------------------> now we print device_placement')
print(sess.run(c))
print('')
##############################

##############################
# Manual device placement
##############################
# Creates a graph.
with tf.device('/cpu:0'):
  a = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], shape=[2, 3], name='a')
  b = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], shape=[3, 2], name='b')
c = tf.matmul(a, b)
# Creates a session with log_device_placement set to True.
sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))
# Runs the op.
print('manual device placement to cpu:0 --------------------------###')
print(sess.run(c))
print('')
##############################

############################
# using single gpu but with allow_soft_placement == True
############################
# Creates a graph.
with tf.device('/device:GPU:2'):
  a = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], shape=[2, 3], name='a')
  b = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], shape=[3, 2], name='b')
  c = tf.matmul(a, b)
# Creates a session with allow_soft_placement and log_device_placement set
# to True.
sess = tf.Session(config=tf.ConfigProto(
      allow_soft_placement=True, log_device_placement=True))
# Runs the op.
print(' using single gpu but with allow_soft_placement == True: /device:GPU:2  ------------------######')
print(sess.run(c))
print('')
############################

#################################
# using a single gpu
#################################
# Creates a graph.
with tf.device('/device:GPU:2'):
  a = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], shape=[2, 3], name='a')
  b = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], shape=[3, 2], name='b')
  c = tf.matmul(a, b)
# Creates a session with log_device_placement set to True.
sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))
# Runs the op.
print('using a single gpu on multi-gpu system: /device:GPU:2 -------------------------###')
print(sess.run(c))
print('')
#################################

######################
# using multiple gpu
######################

# Creates a graph.
c = []
for d in ['/device:GPU:2', '/device:GPU:3']:
  with tf.device(d):
    a = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], shape=[2, 3])
    b = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], shape=[3, 2])
    c.append(tf.matmul(a, b))
with tf.device('/cpu:0'):
  sum = tf.add_n(c)
# Creates a session with log_device_placement set to True.
sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))
# Runs the op.
print('Using multiple GPUs: ---------------------------##')
print(sess.run(sum))
print('')
####################

parser = argparse.ArgumentParser()
parser.add_argument("-i", "--id", help="specify process id")
args = parser.parse_args()
if args.id:
    print('Process ID: %s' % args.id)

import os
import socket
print('Hostname: %s' % socket.gethostname())
print("Hello! Here is some information about the GPU here:")
os.system("nvidia-smi")

