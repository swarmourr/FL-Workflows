#!/usr/bin/env python3

# Model PKG
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import SGD

import glob
from argparse import ArgumentParser


class GlobalModel():

    def __init__(self) -> None:
        pass

     # Build init Models
    def build_model(self,shape, classes) -> None :
        model = Sequential()
        model.add(Dense(100, input_shape=(shape,)))
        model.add(Activation("relu"))
        model.add(Dense(100))
        model.add(Activation("relu"))
        model.add(Dense(classes))
        model.add(Activation("softmax"))
        model.save("global_model.h5")
        return 


    def avg_weights(self,paths,round,global_count=60000):
        '''function for scaling a models weights'''
        scaled_weight_list=[]
        for path in paths: 
            model=load_model(path)
            weight=model.get_weights()
            path_split=path.split(".")[0].split("_")
            scalar=int(path_split[3])-int(path_split[2])/global_count
            weight_final = []
            for i in range(len(weight)):
                weight_final.append(scalar * weight[i])
            scaled_weight_list.append(weight_final)
        '''Return the sum of the listed scaled weights. The is equivalent to scaled avg of the weights'''
        avg_grad = list()
        #get the average grad accross all client gradients
        for grad_list_tuple in zip(*scaled_weight_list):
            layer_mean = tf.math.reduce_sum(grad_list_tuple, axis=0)
            avg_grad.append(layer_mean)
        g_model=load_model(paths[0])
        g_model.set_weights(avg_grad)
        g_model.save(f"global_model_round_{round}.h5")
        return avg_grad
    

if __name__ =="__main__":
    parser = ArgumentParser(description="Local model Federated Learning Workflow")
    parser.add_argument('-r', type=str, help='Round number')
    parser.add_argument('-f', type=str,nargs='+', help='Path to local models', default="")
    args = parser.parse_args()
    # print(args.f)
    # paths=glob.glob(args.f+"local_model_[0-9]*_*[0-9].h5")
    # print(paths)
    GlobalModel().avg_weights(args.f,args.r)
   

