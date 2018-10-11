import numpy as np
import pandas as pd
import librosa
import os
from python_speech_features import fbank
from multiprocessing import Pool,cpu_count
import argparse
import traceback
import os

parser = argparse.ArgumentParser(description='npy file creator')
parser.add_argument('--csv', type=str, help='Input csv filpath')
parser.add_argument('--partition', type=int, help='Number of splits to make',default=128)
parser.add_argument('--seg_len', type=int, help='signal length with Zero padding',default=3)
parser.add_argument('--rate', type=int, help='Sampling Rate',default=16000)
parser.add_argument('--keep_wav', type=str, help='"T" if the wav files has to be kept else "F"',default="F")

args = parser.parse_args()
SAMPLE_RATE=args.rate
csv_filepath=args.csv
seg_len=args.seg_len




def normalize_frames(m):
    return [(v - np.mean(v)) / (np.std(v)+1e-12) for v in m]
def read_audio(filename, sample_rate=SAMPLE_RATE):
    audio, sr = librosa.load(filename, sr=SAMPLE_RATE, mono=True)
    audio = audio.flatten()
    return audio
def pre_process_inputs(filename, target_sample_rate=SAMPLE_RATE,seg_len=seg_len):
    if not os.path.exists(filename):
        return False  
    signal=read_audio(filename)
    signal=signal[:(SAMPLE_RATE*seg_len)]
    padding_len=(SAMPLE_RATE*seg_len)-len(signal)
    signal=np.concatenate((signal,np.array([0.0]*padding_len)))
    filter_banks, energies = fbank(signal, samplerate=target_sample_rate, nfilt=64, winlen=0.025)
    filter_banks = normalize_frames(filter_banks)
    flatten=[]
    for i in filter_banks:
        flatten.extend(list(i))
    np.save(filename.replace('.wav', '.npy'),flatten)
    if args.keep_wav.upper()=='F':
        os.remove(filename)
    return True
def pre_process_inputs_MP(data_split):
        return np.array(data_split.apply(pre_process_inputs).tolist())
    
if __name__ == '__main__':
    try:
        print('Processing!')
        data=pd.read_csv(csv_filepath)
        data_split=np.array_split(data['filename'],int(args.partition))
        pool=Pool(cpu_count())
        x=np.concatenate(pool.map(pre_process_inputs_MP,data_split))
        pool.close()
        pool.join()
    except Exception as err:
        print(traceback.format_exc())

    

