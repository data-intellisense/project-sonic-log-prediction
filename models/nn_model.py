from keras.utils import to_categorical
from keras.models import Sequential
# from keras.models import model
from keras.layers.core import Dense, Activation, Dropout
from keras.optimizers import SGD, Adam
from keras.layers.normalization import BatchNormalization
from keras.layers.core import  Dropout
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers.core import Flatten
from keras.callbacks import EarlyStopping
from keras.callbacks import ModelCheckpoint
from keras.optimizers import SGD, Adam,RMSprop,Adagrad,Adadelta,Nadam # optimization algorithms
from keras.callbacks import ModelCheckpoint

class nn_model():
    def __init__(self):
        return None
    
    def fit(self, X_train, y_train, sample_weight=None):
        
        # hyper parameters
        NB_EPOCH = 150 # number of epoch
        BATCH_SIZE = 30 # mini batch size
        VERBOSE = 1 #display results during training
        NB_CLASSES = 10 #number of classes
        OPTIMIZER = SGD() # choose optimizer
        N_HIDDEN = 128 # number of nodes in the hidden layer
        VALIDATION_SPLIT = 0.2 #80% training and 20%validation
        METRICS =['accuracy']
        LOSS = 'categorical_crossentropy' # because of multiclass
        
        filepath="models/best_nn_model.hdf5"

        self.model = Sequential()
        self.model.add(Dense(1,input_shape=(N_HIDDEN,)))
        self.model.add(Activation('sigmoid'))
        print(self.model.summary())

        early_stopping_monitor = EarlyStopping(monitor='val_loss',patience = 8)                
        checkpoint = ModelCheckpoint(filepath,monitor='val_loss',verbose=1,save_best_only=True)
        
        self.model.compile(loss=LOSS, optimizer = OPTIMIZER, metrics =METRICS)
        self.model.fit(X_train, y_train, sample_weight=sample_weight,
                batch_size=BATCH_SIZE, epochs = NB_EPOCH, verbose = VERBOSE,
                validation_split = VALIDATION_SPLIT,callbacks=[checkpoint,early_stopping_monitor])

    def predict(self, X_test):
        
        return self.model.predict(X_test)




