#from batchprocess import Batch_data, pretraining_batch
import tensorflow as tf
import pandas as pd
import itertools
import os
import logging
import datetime
import argparse
import numpy as np
from collections import Counter
import time
from nets import model
from batch_triplet import MultiTaskBatchManager
from tools import get_available_gpus,average_gradients
import multiprocessing as mp


parser = argparse.ArgumentParser(description='Model Training')
parser.add_argument('--chkpt', type=str, help='checkpoint dir name')
parser.add_argument('--chkpt_load_dir', type=str, help='pre trained checkpoint dir name',default=' ')
parser.add_argument('--E_train', type=int, help='number of training epochs',default=5)
parser.add_argument('--lr', type=float, help='learning rate',default=0.0001)
parser.add_argument('--bs', type=int, help='training BatchSize',default=64)


args = parser.parse_args()
parameters='--chkpt '+args.chkpt+'--chkpt_load_dir '+args.chkpt_load_dir
parameters+='--E_train '+str(args.E_train)+'--lr '+str(args.lr)+'--bs '+str(args.bs)
 
if args.chkpt_load_dir==' ':
    chkpt_load_dir=args.chkpt
else:
    chkpt_load_dir=args.chkpt_load_dir
chkpt_dir = args.chkpt
if not os.path.exists(chkpt_dir):
    os.mkdir(chkpt_dir)

train_df = pd.read_csv('ted_train.csv')
total_speakers = max(train_df['speaker_id'].unique())
batch_size=args.bs
    
    

logging.basicConfig(
    filename=chkpt_dir + '/' + datetime.datetime.now().strftime('training_log_%d_%m_%Y_%H_%M.log'),
    level=logging.DEBUG)
logging.info(datetime.datetime.now().strftime('%d_%m_%Y_%H_%M_') + 'Training Parameters : %s' %parameters)
chkpt_file=chkpt_dir + '/model.ckpt'

T_graph=tf.Graph()
num_epoch=args.E_train

def triplet_loss(_input, alpha):
    anchor, positive, negative = tf.split(_input, 3, axis=0)
    with tf.variable_scope('triplet_loss'):
        mul_ap = tf.reduce_sum(tf.multiply(anchor, positive), axis=1)
        mul_an = tf.reduce_sum(tf.multiply(anchor, negative), axis=1)
        mod_ap = tf.multiply(tf.sqrt(tf.reduce_sum(tf.square(anchor), axis=1)),
                             tf.sqrt(tf.reduce_sum(tf.square(positive), axis=1)))
        mod_an = tf.multiply(tf.sqrt(tf.reduce_sum(tf.square(anchor), axis=1)),
                             tf.sqrt(tf.reduce_sum(tf.square(negative), axis=1)))

        pos_dist = tf.divide(mul_ap, mod_ap)
        neg_dist = tf.divide(mul_an, mod_an)
        basic_loss = tf.add(tf.subtract(neg_dist, pos_dist), alpha)
        loss = tf.reduce_mean(tf.maximum(basic_loss, 0.0), 0)  # tf.reduce_mean
    return loss

def train():
    BatchLoader=MultiTaskBatchManager(Data=train_df,batch_size=batch_size,Ntasks=5,Nepochs=num_epoch)
    with T_graph.as_default():
        tower_grads=[]
        global_step = tf.Variable(0, name='global_step', trainable=False)
        x=tf.placeholder('float32')
        alpha=tf.placeholder('float32')
        optimizer = tf.train.AdamOptimizer(learning_rate=args.lr)
        available_gpus=get_available_gpus()
        num_clones=len(available_gpus)
        print('Number of clones = %d'%num_clones)
        with tf.variable_scope(tf.get_variable_scope()):
            for i in range(num_clones):
                with tf.device(available_gpus[i]):
                    # Network outputs.
                    prediction= model(x[i],batch_size*3,total_speakers)
                    prediction=tf.nn.l2_normalize(prediction,1, 1e-10, name='embeddings')
                    with tf.name_scope('loss'):
                        loss =triplet_loss(prediction, alpha)
                    # Reuse variables for the next tower.
                    tf.get_variable_scope().reuse_variables()
                    # Calculate the gradients for the batch of data on this tower.
                    grads = optimizer.compute_gradients(loss)
                    tower_grads.append(grads)
        grads = average_gradients(tower_grads)
        # Apply the gradients to adjust the shared variables.
        apply_gradient_op = optimizer.apply_gradients(grads, global_step=global_step)
        # Track the moving averages of all trainable variables.
        MOVING_AVERAGE_DECAY = 0.9999
        variable_averages = tf.train.ExponentialMovingAverage(MOVING_AVERAGE_DECAY, global_step)
        variables_averages_op = variable_averages.apply(tf.trainable_variables())
        train_op = tf.group(apply_gradient_op, variables_averages_op)

        summaries=set(tf.get_collection(tf.GraphKeys.SUMMARIES))
        summaries.add(tf.summary.scalar('loss',loss))
        summary_op=tf.summary.merge(list(summaries))
        
        with tf.Session(graph=T_graph,config=tf.ConfigProto(allow_soft_placement=True,log_device_placement=False)) as sess:
            saver = tf.train.Saver()
            summary_writer=tf.summary.FileWriter(chkpt_dir,graph=T_graph)
            sess.run(tf.global_variables_initializer())
            tf.train.start_queue_runners(sess=sess)
            if os.path.exists(chkpt_dir + '/checkpoint'):
                print('restoring !!')
                saver.restore(sess, chkpt_file)
            elif not os.path.exists(chkpt_dir):
                os.mkdir(chkpt_dir)
            print('Training Started !!')
            isrunning=True
            stepcount=0
            steploss=0
            while isrunning:
                stepcount+=1
                #st=time.time()
                batch_xs=[]
                for _ in range(num_clones):
                    batch_x, batch_y, flag,isrunning=BatchLoader.next_batch()
                    batch_xs.append(batch_x)
                    if not isrunning:
                        break;
                if not isrunning:
                    break;
                #print('data loading time %s'%(time.time()-st))
                if isrunning:
                    #st=time.time()
                    _,c,summary,g= sess.run([apply_gradient_op,loss,summary_op,global_step],feed_dict={x:batch_xs,alpha: 0.1})
                    #print('training time %s'%(time.time()-st))
                    steploss+=c
                if stepcount%100==0:
                    save_path = saver.save(sess, chkpt_file)
                    print('step_loss : %s '%steploss)
                    steploss=0
            BatchLoader.close()
    
    
if __name__ == '__main__':
    train()
