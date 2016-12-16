__author__ = "Mehdi Korjani"
__version__ = "1.0.0"

import pdb
import glob
import os
import numpy as np
from keras.models import load_model
from keras.models import model_from_json
from keras.callbacks import EarlyStopping
from keras.callbacks import ModelCheckpoint
import h5py
import argparse
from pydub import AudioSegment
import sys
sys.path.append(os.path.abspath('utils'))
import preprocessing as frame
import wave_manipulation as manipulate
import generate_model

MODEL_FILE = 'model/model_weights_11_21_16.h5'

SPEECH_FILE = '/home/mehdi/data/corpus/VCTK-Corpus/wav48/*/*.wav'
NOISE_FILE = '/home/mehdi/data/corpus/office_noise/*.wav'
RESULT_DIR = 'model/model_12_8_16.h5'


NB_EPOCH = 100000
BATCH_SIZE = 200

FRAME_STEP = 50
FRAME_LEN = 6500  

MIN_SNR = 10
MAX_SNR = 12

X_train = np.array([])
Y_train = np.array([])  

def get_arguments():
    parser = argparse.ArgumentParser(description='train Deep Learning model')
    parser.add_argument('--model_file', type=str, default=MODEL_FILE,
                        help='The directory containing the file model .')
    parser.add_argument('--speech_file', type=str, default=SPEECH_FILE,
                        help='Speech files.')
    parser.add_argument('--noise_file', type=str, default=NOISE_FILE,
                        help='Noise files.')
    parser.add_argument('--nb_epoch', type=int, default=NB_EPOCH,
                        help='number of epoch')
    parser.add_argument('--batch_size', type=int, default=BATCH_SIZE,
                        help='batch size')
    parser.add_argument('--min_snr', type=int, default=MIN_SNR,
                        help='minimum of SNR signal to noise')
    parser.add_argument('--max_snr', type=int, default=MAX_SNR,
                        help='maximum of SNR signal to noise')                        
    parser.add_argument('--frame_step', type=int, default=FRAME_STEP,
                        help='frame step')
    parser.add_argument('--frame_len', type=int, default=FRAME_LEN,
                        help='frame length')                        
    parser.add_argument('--result_dir', type=str, default=RESULT_DIR,
                        help='directory to write result.')
    return parser.parse_args()

def validate_arg(arg):
    try:
        os.path.isfile(arg)
        print('file exists: %s' %arg)
    except:
        print('file not found: %s' %arg)
        return
    try:
        os.access(arg, os.R_OK)
        print('file is readable: %s' %arg)
    except:
        print('file is not readable: %s' %arg)
        return
    
def data_genertor():
    args = get_arguments()  
    frame_len =args.frame_len
    frame_step = args.frame_step
    while True: 
        
        for fullpath in glob.iglob(args.speech_file):
            fs_signal, signal_sound_data = manipulate.wavread(fullpath)
            signal_sound = AudioSegment.from_file(fullpath)

            for fullpath_noise in glob.iglob(args.noise_file): 
                fs_noise, noise_sound_data = manipulate.wavread(fullpath_noise)
                noise_sound = AudioSegment.from_file(fullpath_noise)
                
                SNR = np.random.randint(args.min_snr,args.max_snr)
                dB = signal_sound.dBFS - noise_sound.dBFS - SNR
                noise_sound += dB # adjust dB for noise relative to sound
                noise_sound_data = noise_sound.get_array_of_samples() 
                
                rand_start = np.random.randint(len(noise_sound_data)- len(signal_sound_data))
                # check the lenght of signal and noise , assume len(noise) > len(signal)

                combined = signal_sound_data + noise_sound_data[rand_start: rand_start+ len(signal_sound_data)]
                noisy_data = combined.astype(np.int16)
                
                # nosrmalized data [0,1]
                noisy_data_norm = manipulate.normalize(noisy_data)
                signal_sound_data_norm = manipulate.normalize(signal_sound_data)
  
                framed_noisy =  frame.framesig(noisy_data_norm,frame_len,frame_step) 
                framed_clean =  frame.framesig(signal_sound_data_norm,frame_len,frame_step)

                #in_out =np.hstack((framed_noisy, framed_clean))
                #np.random.shuffle(in_out)
                #X_train = in_out[:,:frame_len]
                #audio = in_out[:,frame_len + frame_len/2]
                X_train = framed_noisy
                audio = framed_clean[:,frame_len/2]
                
                ulaw_audio = frame.ulaw(audio)
                digit_audio = frame.float_to_uint8(ulaw_audio)
                Y_train = frame.one_hot(digit_audio)
                   
                yield X_train , Y_train # yield

                
                
                
def main():
    args = get_arguments()

    ## read model
    try:
        model = generate_model.generate()
        model.summary()
    except:
        print('cant read the model!' )
        return
        
    ## load weights
    try:
        model.load_weights(args.model_weight)
    except:
        print('no weight!' )
        
    ## validate wave file 
    try:
        validate_arg(args.speech_file)
    except ValueError as e:
        print("wave file is not available:")
        print(str(e))
        return        
        
    data = data_genertor()
    
    ## training
    checkpoint = ModelCheckpoint(args.result_dir, monitor='val_acc', verbose=0, save_weights_only=False, save_best_only=False , mode='auto') 
    callbacks_list = [checkpoint]  
    nb_files =40000         
    print('Training model...')

    model.fit_generator(data,
                        nb_files,
                        nb_epoch=args.nb_epoch,
                        callbacks=callbacks_list,
                        verbose = 1)

                        
 
if __name__ == '__main__':
    main()                     
                    