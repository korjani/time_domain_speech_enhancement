__author__ = "Mehdi Korjani"
__version__ = "1.0.0"


from keras.models import load_model
import simplejson
import pdb 
import cPickle as pickle
import os
import h5py
import argparse
sys.path.append(os.path.abspath('utils'))
import preprocessing as frame
import wave_manipulation as manipulate

MODEL_FILE = 'model/model_weights_11_21_16_SNR_time_domain_time_domain_mlp_6500_4500_3500_2500_1500_256.h5'
SPEECH_FILE = '/home/mehdi/data/corpus/VCTK-Corpus/wav48/p228/p228_145.wav'
NOISE_FILE =  '/home/mehdi/data/corpus/office_noise/street.wav' 
RESULT_DIR = 'result/'
SNR = 15
frame_step = 1
FRAME_LENGHT = 6500
 
def get_arguments():
    parser = argparse.ArgumentParser(description='read h5py model and save the model in a json file and weights in a pickle file')
    parser.add_argument('--model_file', type=str, default=MODEL_FILE,
                        help='The directory containing the h5df model (model + weights).')
    parser.add_argument('--noisy_file', type=str, default=False,
                        help='noisy file.')
    parser.add_argument('--speech_file', type=str, default=SPEECH_FILE,
                        help='Speech file.')
    parser.add_argument('--noise_file', type=str, default=NOISE_FILE,
                        help='noise file.')
    parser.add_argument('--snr', type=int, default=SNR,
                        help='SNR of noisy data (how much noise is added to data.')
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
    
    

def main():
    args = get_arguments()
    
    ## check model argument
    try:
        validate_arg(args.model_file)
    except ValueError as e:
        print("some arguments are wrong:")
        print(str(e))
        return  

    ## read model
    try:
        model =load_model(args.model_file)
        model.summary()
    except: 
        print('cant read the model')
        return
    
    ## validate wave file 
    try:
        validate_arg(args.speech_file)
        validate_arg(args.noise_file)
    except ValueError as e:
        print("wave file is not available:")
        print(str(e))
        return          
    
    ## read wavefile
    fs_signal, signal_sound_data = manipulate.wavread(args.speech_file)
    fs_noise, noise_sound_data = manipulate.wavread(args.noise_file)
    signal_sound = AudioSegment.from_file(args.speech_file)
    noise_sound = AudioSegment.from_file(args.noise_file)
    
    dB = signal_sound.dBFS - noise_sound.dBFS - args.snr
    noise_sound += dB # adjust dB for noise relative to sound
    noise_sound_data = noise_sound.get_array_of_samples()
                    
    rand_start = np.random.randint(len(noise_sound_data)- len(signal_sound_data))
    ## check the lenght of signal and noise , assume len(noise) > len(signal)
    ## add noise or check clean data
    combined = signal_sound_data + noise_sound_data[rand_start: rand_start+ len(signal_sound_data)]

    noisy_data = combined.astype(np.int16)
    # normalized data [0,1]
    noisy_data_norm = manipulate.normalize(noisy_data)
    len_data = len(noisy_data_norm)
    Y_arg = np.array([])
    nb_batch = 5
    for i in range(nb_batch):
        print(i)
        X_train =  frame.framesig(noisy_data_norm[max(i*(len_data/nb_batch)-frame_len+1,0):(i+1)*(len_data/nb_batch)],frame_len,frame_step) 
        Y_pred = model.predict(X_train)
        Y_arg_batch = np.argmax(Y_pred,axis=1)
        Y_arg = np.append(Y_arg, Y_arg_batch)

    Y_ = frame.ulaw2lin(Y_arg)
    
    Y_ = manipulate.denormalize(Y_)
    ## same lenght as input wave
    Y = np.append(np.zeros(frame_len/2),Y_ )
    Y = np.append(Y, np.zeros(frame_len/2) )
        


    wavwrite(args.result_dir + str(SNR)+'_' + tail[:-4] + '_' + tail_noise[:-4] + '.wav', fs_signal, Y)
    noisy_data = np.array(noisy_data,dtype=np.int16)
    wavwrite('/data1/mehdi/data/test/time_domain_results/noisy_SNR_'+str(SNR)+'_' + tail[:-4] + '_' + tail_noise[:-4] + '.wav', fs_signal, noisy_data)
    #pdb.set_trace()
    Y_new = Y
    moving_range  = 6
    for i in range(moving_range/2, len(Y)-moving_range/2):
        Y[i] = np.mean(Y_new[i-moving_range/2:i+moving_range/ 2])

    Y = np.asarray(Y,dtype=np.int16)
    wavwrite('/data1/mehdi/data/test/time_domain_results/result_SNR_'+str(SNR)+'_' + tail[:-4] + '_' + tail_noise[:-4] + '_moving_average.wav', fs_signal, Y)

    wavwrite('/data1/mehdi/data/test/time_domain_results/signal' +'_' + tail[:-4] + '_' + tail_noise[:-4] + '.wav', fs_signal, data)
    
    
    
 
if __name__ == '__main__':
    main() 