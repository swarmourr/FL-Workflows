#!/usr/bin/env python3


# Data  PKG
from argparse import ArgumentParser
import idx2numpy
from sklearn.preprocessing import LabelBinarizer 

# Model PKG
from tensorflow.keras.models import load_model
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import SGD

class LocalTraining:

    def __init__(self):
        self.X_train=None
        self.Y_train=None
        # Optimiser
        learning_rate = 0.01 
        self.loss='categorical_crossentropy'
        self.metrics = ['accuracy']
        self.optimizer = SGD(learning_rate=learning_rate,momentum=0.9) 
           
    def processdata(self,start:int,end:int)-> None:
        self.start=start
        self.end=end
        self.X_train = idx2numpy.convert_from_file('train-images-idx3-ubyte')[start:end]
        self.Y_train = idx2numpy.convert_from_file('train-labels-idx1-ubyte')[start:end]
        print(type(self.X_train))
        self.X_train = self.X_train.reshape(self.X_train.shape[0], -1) 
        self.X_train = self.X_train.astype('float32')
        self.X_train /= 255
        lb = LabelBinarizer()
        self.Y_train = lb.fit_transform(self.Y_train)

  
    def localtrain(self,model,round)-> None:
        local_model=load_model(model)
        local_model.compile(loss=self.loss,optimizer=self.optimizer,metrics=self.metrics)
        local_model.fit(self.X_train,self.Y_train,epochs=2,batch_size=200)
        local_model.save(f"local_model_{self.start}_{self.end}_round_{round}.h5")
        return 
    
   
if __name__ == '__main__':
    parser = ArgumentParser(description="Local model Federated Learning Workflow")
    parser.add_argument('-s', type=int,  help='Starting point of dataset')
    parser.add_argument('-e', type=int,  help='Ending point of dataset')
    parser.add_argument('-r', type=int,  help='Round number')
    parser.add_argument('-m', type=str,  help='Model name')
    args = parser.parse_args()
    model=LocalTraining()
    model.processdata(args.s,args.e)
    model.localtrain(args.m,args.r)
