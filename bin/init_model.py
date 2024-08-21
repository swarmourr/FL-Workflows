#!/usr/bin/env python3

# Model PKG

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import Dense

from argparse import ArgumentParser


class InitGlobalModel():

    def __init__(self) -> None:
        pass

     # Build init Models
    def build_model(self,shape, classes , name) -> None :
        model = Sequential()
        model.add(Dense(100, input_shape=(shape,)))
        model.add(Activation("relu"))
        model.add(Dense(100))
        model.add(Activation("relu"))
        model.add(Dense(classes))
        model.add(Activation("softmax"))
        model.save(name)
        return



if __name__ =="__main__":
    parser = ArgumentParser(description="Buil the initial global model for Federated Learning Workflow")
    parser.add_argument('-n', type=str, help='Name of global model', default="global_model_round_init.h5")
    parser.add_argument('-s', type=int, help='shape of Neural Network', default=784)
    parser.add_argument('-c', type=str, help='Number of classes', default=10)
    args = parser.parse_args()
    InitGlobalModel().build_model(args.s,args.c,args.n)
