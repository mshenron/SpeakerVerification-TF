import pandas as pd
import numpy as np
import time
from multiprocessing import Pool, cpu_count,Process, Queue, Event
import logging
import datetime
import os
import itertools
from sklearn.metrics.pairwise import cosine_similarity
from nets import model
import tensorflow as tf 

def load_filterbanks(filename):
    return np.load(filename.replace('.wav','.npy'))

def load_filterbanks_mp(data_split):
    return [np.load(i.replace('.wav','.npy')) for i in data_split]
class Batch_data():
    def __init__(self, Batch, BatchAP, MiniBatchSize,chkpt_dir):
        self.MiniBatchSize = MiniBatchSize
        self.Batch = Batch
        ap_pairs = []
        for spk in self.Batch['speaker_id'].unique():
            files = list(self.Batch[self.Batch['speaker_id'] == spk]['filename'])
            for i in itertools.combinations(files, 2):
                ap_pairs.append([i[0], i[1], spk])
        self.BatchAP = pd.DataFrame(ap_pairs, columns=['anchor', 'positive', 'speaker_id'])
        self.Cores = cpu_count()/2
        self.Partitions = min(self.Cores, MiniBatchSize * 3)
        self.Speakers = self.Batch['speaker_id'].unique().tolist()
        self.APlen = np.inf
        
        embeddings=model(x,1)
        tf.get_variable_scope().reuse_variables()
        saver=tf.train.Saver()
        with  tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            if os.path.exists(chkpt_dir+'/checkpoint'):
                saver.restore(sess, chkpt_dir+'/model.ckpt')
                print('restoring !')
            else:
                print('Checkpoint File Not Found !')
            xs=self.Batch['filename'].apply(lambda xx:load_filterbanks(xx))
            filenames=self.Batch['filename'].tolist()
            embds={}
            for _x in range(xs):
                embds[filenames[_x]]=sess.run(embeddings,feed_dict={x:xs[_x]})
        self.embds=embds

    def to_inputs(self):
        data_split = np.array_split(self.MiniBatch,self.Partitions)
        pool = Pool(self.Cores)
        x = np.concatenate(pool.map(load_filterbanks_mp, data_split))
        pool.close()
        pool.join()
        flag=len(self.BatchAP) > self.MiniBatchSize
        return x, None, flag

    def GetMiniBatch(self):
        if not len(self.BatchAP) > self.MiniBatchSize:
            return None, None, False
        AP_pair = self.BatchAP.sample(n=self.MiniBatchSize)
        BatchAPIndex = list(AP_pair.index)
        
        AP_pair['negative'] = AP_pair['speaker_id'].apply(
            lambda x: self.Batch[self.Batch['speaker_id'] != x].sample(n=1)['filename'].tolist()[0])
        self.BatchAP = self.BatchAP.drop(BatchAPIndex, axis=0)
        self.MiniBatch=np.concatenate((AP_pair['anchor'].tolist(), AP_pair['positive'].tolist(), AP_pair['negative'].tolist()))
        return (self.to_inputs())

class SingleBatchGenrator(Process):
    def __init__(self,single_task_q,stop_event,Data,batch_size,MiniBatchSize,Nepochs,seed,chkpt_dir):
        super(SingleBatchGenrator,self).__init__()
        self.done_q=single_task_q
        self.chkpt_dir=chkpt_dir
        self.stop_event=stop_event
        self.Data=Data
        self.seed=seed
        self.batch_size=batch_size
        self.MiniBatchSize=MiniBatchSize
        self.train_data=Batch_data(Batch=self.Data,MiniBatchSize=self.MiniBatchSize,self.chkpt_dir)
        self.Nepochs=Nepochs
        self.Depochs=0
        self.count=0
    def next_batch(self):
        return self.train_data.GetMiniBatch()
    def run(self):
        while self.Depochs<=self.Nepochs and not self.stop_event.is_set():
            
            flag=True
            while flag and not self.stop_event.is_set():
                if not self.done_q.full():
                    x, y, flag=self.next_batch()
                    self.done_q.put([x, y, flag])
                    self.count+=1
                else:
                    time.sleep(0.01)
            self.Depochs+=1
            self.train_data=Batch_data(Batch=self.Data,BatchAP=self.ap_pairs,MiniBatchSize=self.MiniBatchSize,self.chkpt_dir)
            
        print('event stopped! count :%d'%self.count )
        self.stop_event.set()
            
class BatchAggregator(Process):
    def __init__(self,single_task_qs,multi_task_q ,stop_event,single_task_stop_events):
        super(BatchAggregator,self).__init__()
        self.pending_qs=single_task_qs
        self.done_q=multi_task_q
        self.stop_event=stop_event
        self.single_task_stop_events=single_task_stop_events
    def run(self):
        while not self.stop_event.is_set():
            if not self.done_q.full():
                for pending_q in self.pending_qs:
                    if not pending_q.empty():
                        x, y, flag=pending_q.get()
                        self.done_q.put([x, y, flag,True])
            if self.done_q.empty():
                events=[e.is_set() for e in self.single_task_stop_events]
                if all(events):
                    self.done_q.put([None,None,None,False])
                
class MultiTaskBatchManager:
    def __init__(self,Data,batch_size,MiniBatchSize,Ntasks,Nepochs,checkpoint_dir):
        MAX_CAPACITY=Ntasks*10
        self.chkpt_dir=checkpoint_dir
        self.batch_size=batch_size
        self.MiniBatchSize=MiniBatchSize
        self.Ntasks=Ntasks
        self.Nepochs=Nepochs
        self.stop_event=Event()
        self.single_task_stop_events=[]
        self.single_task_qs=[]
        for _ in range(Ntasks):
            self.single_task_qs.append(Queue(MAX_CAPACITY))
            self.single_task_stop_events.append(Event())
        self.multi_task_train_q=Queue(MAX_CAPACITY*2)
        self.BatchAggregator=BatchAggregator(self.single_task_qs,self.multi_task_train_q,self.stop_event,self.single_task_stop_events)
        self.Data=Data#list(range(5))
        self.step=int(len(self.Data)/Ntasks)
        self.batch_generator={i: SingleBatchGenrator(self.single_task_qs[i],self.single_task_stop_events[i],self.Data[i*self.step:(i+1)*self.step],self.batch_size,self.MiniBatchSize,self.Nepochs,i,self.chkpt_dir) for i in range(Ntasks)}
        for w in self.batch_generator.values():
            w.start()
        self.BatchAggregator.start()
        self.isalldone=False
    
    def next_batch(self):
        slept=0
        while True:
            if not self.multi_task_train_q.empty():
                x, y, flag,isrunning=self.multi_task_train_q.get()
                if not isrunning:
                    self.isalldone=True
                return([x, y, flag,isrunning])
            if self.isalldone:
                return([None,None,None,False])
            time.sleep(0.01)
            slept+=1
            if slept>10000:
                print('Error Fetching Batch!')
                return([None,None,None,False])
            
        
    def close(self,timeout=5):
        self.stop_event.set()
        for se in self.single_task_stop_events:
            se.set()
        for w in self.batch_generator.values():
            w.join(timeout=timeout)
            w.terminate()
        self.BatchAggregator.join(timeout=timeout)
        self.BatchAggregator.terminate()
        
        
        