from keras.models import load_model
import simplejson
import pdb 
import cPickle as pickle
import os
import h5py

hdf_file = '../model/model_weights_11_21_16_SNR_time_domain_time_domain_mlp_6500_4500_3500_2500_1500_256_bk.h5'

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
    try:
        validate_arg(hdf_file)
    except ValueError as e:
        print("some arguments are wrong:")
        print(str(e))
        return  
            
    ## read model
    try:
        model =load_model(hdf_file)
        model.summary()
    except: 
        print('cant read the model')
        return      

    model_json = model.to_json()
    with open(hdf_file[:-3] + '.json', 'w') as json_file:
        json_file.write(simplejson.dumps(simplejson.loads(model_json), indent = 4, sort_keys=True))
        
    weight = model.get_weights()
    pickle.dump(weight, open(hdf_file[:-3] + '.pkl', 'wb' ) )
 
if __name__ == '__main__':
    main()                   