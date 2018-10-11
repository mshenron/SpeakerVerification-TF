import pandas as pd
import numpy as np
import time
from multiprocessing import Pool, cpu_count,Process, Queue, Event

#def load_filterbanks(filename):
#    return np.load(filename.replace('.wav', '.npy'))
#def load_filterbanks_mp(data_split):
#    return np.array(data_split.apply(load_filterbanks).tolist())

def load_filterbanks_mp(data_split):
    return [np.load(i.replace('.wav','.npy')) for i in data_split]
class TrainData():
    def __init__(self,Data,BatchSize):
        self.BatchSize=BatchSize
        self.Cores = 10#cpu_count()/2
        self.Partitions=min(self.Cores, BatchSize)
        self.Data=Data
    def GetBatch(self):
        BatchData=self.Data.sample(self.BatchSize)
        BatchIndex = list(BatchData.index)
        self.Data=self.Data.drop(BatchIndex, axis=0)
        BatchData_filenames=np.array(BatchData['filename'])
        data_split = np.array_split(BatchData_filenames, self.Partitions)
        pool = Pool(self.Cores)
        x = np.concatenate(pool.map(load_filterbanks_mp, data_split))
        y = np.array(BatchData['speaker_id'].tolist())
        pool.close()
        pool.join()
        flag = len(self.Data) > self.BatchSize
        return x, y, flag


class SingleBatchGenrator(Process):
    def __init__(self,single_task_q,stop_event,Data,batch_size,Nepochs,seed):
        super(SingleBatchGenrator,self).__init__()
        self.done_q=single_task_q
        self.stop_event=stop_event
        self.Data=Data
        self.seed=seed
        self.batch_size=batch_size
        self.train_data=TrainData(self.Data,self.batch_size)
        self.Nepochs=Nepochs
        self.Depochs=0
        self.count=0
    def next_batch(self):
        return self.train_data.GetBatch()
    def run(self):
        while self.Depochs<self.Nepochs and not self.stop_event.is_set():
            flag=True
            while flag and not self.stop_event.is_set():
                if not self.done_q.full():
                    x, y, flag=self.next_batch()
                    self.done_q.put([x, y, flag])
                    self.count+=1
                else:
                    time.sleep(0.01)
            self.Depochs+=1
            self.train_data=TrainData(self.Data,self.batch_size)
        print('SingleBatchGenrator_%s event stopped! count :%d'%(self.seed,self.count) )
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
    def __init__(self,Data,batch_size=64,Ntasks=5,Nepochs=1):
        MAX_CAPACITY=Ntasks*10
        self.batch_size=batch_size
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
        self.batch_generator={i: SingleBatchGenrator(self.single_task_qs[i],self.single_task_stop_events[i],self.Data[i*self.step:(i+1)*self.step],self.batch_size,self.Nepochs,i) for i in range(Ntasks)}
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
        
        
        