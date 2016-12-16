from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.optimizers import SGD, RMSprop,Adam

def generate():
    layer1_dimention = 4500
    layer2_dimention = 3500
    layer3_dimention = 3000
    layer4_dimention = 2000
    
    nb_epoch = 100000
    batch_size = 100

    frame_step = 10
    frame_len = 6500
    input_dim = frame_len
    output_dim = 256
    
    model = Sequential()
    model.add(Dense(input_dim= input_dim, output_dim = layer1_dimention, init='glorot_uniform' ) )
    model.add(Activation('relu'))
    model.add(Dense(input_dim=layer1_dimention, output_dim = layer2_dimention, init='glorot_uniform' ) )
    model.add(Activation('relu'))
    model.add(Dense(input_dim=layer1_dimention, output_dim = layer3_dimention, init='glorot_uniform' ) )
    model.add(Activation('relu'))
    model.add(Dense(input_dim=layer1_dimention, output_dim = layer4_dimention, init='glorot_uniform' ) )
    model.add(Activation('relu'))
    model.add(Dense(input_dim=layer4_dimention, output_dim = output_dim, init='glorot_uniform' ))
    model.add(Activation('softmax'))

    RMS = RMSprop(lr=0.001, rho=0.9, epsilon=1e-12)
    model.compile(loss='categorical_crossentropy', optimizer=RMS, metrics=['accuracy'])
    model.summary() 
    
    
    return model

if __name__=='__main__':
    pass