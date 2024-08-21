#!/usr/bin/env python3

# Model PKG
from argparse import ArgumentParser
import datetime
import idx2numpy
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import load_model

from sklearn.preprocessing import LabelBinarizer 
from sklearn.metrics import accuracy_score
import tensorflow as tf

class ModelEvaluation():

    def __init__(self,model_path):
       self.model=load_model(model_path)
        
        
        
    def processdata(self,start,end):
        self.start=start
        self.end=end
        self.X_test = idx2numpy.convert_from_file('t10k-images-idx3-ubyte')[start:end]
        self.Y_test = idx2numpy.convert_from_file('t10k-labels-idx1-ubyte')[start:end]
        self.X_test = self.X_test.reshape(self.X_test.shape[0], -1) 
        self.X_test = self.X_test.astype('float32')
        self.X_test /= 255
        lb = LabelBinarizer()
        self.Y_test = lb.fit_transform(self.Y_test)
    
    def evaluation(self,client,round):
        cce = tf.keras.losses.CategoricalCrossentropy(from_logits=True)
        logits = self.model.predict(self.X_test)
        loss = cce(self.Y_test, logits)
        acc = accuracy_score(tf.argmax(logits, axis=1), tf.argmax(self.Y_test, axis=1))
        tmp_eval_g={'id':int(client),'acc':float(acc)*100,'loss':float(loss),'client': f"local_model_{self.start}_{self.end}","type":"global"}
        perfermence_df=pd.DataFrame([tmp_eval_g])
        perfermence_df.to_csv(f"global_model_evaluation_{client}_{round}.csv")
        return  acc, loss
    

if __name__ =="__main__":
    #eval_model=ModelEvaluation()
    parser = ArgumentParser(description="Local model Federated Learning Workflow")
    parser.add_argument('-s', type=int,  help='Starting point of dataset')
    parser.add_argument('-e', type=int,  help='Ending point of dataset')
    parser.add_argument('-c', type=int,  help='Ending point of dataset')
    parser.add_argument('-m', type=str,  help='Model name')
    parser.add_argument('-r', type=int,  help='Round number')
    args = parser.parse_args()
    eval_model=ModelEvaluation(args.m)
    eval_model.processdata(int(args.s),int(args.e))
    eval_model.evaluation(args.c,args.r)
