#!/usr/bin/env python3

import pandas as pd 
import numpy as np 
from argparse import ArgumentParser

# --- Import Pegasus API -----------------------------------------------------------
from Pegasus.api import *

class Noop:
    
   def __init__(self) -> None:
      pass
             
   def logs(self,round,score,name):
      print(name)
      f = open(name, "w+")
      f.write(f"The Federted learning Traiing finished after executing {round} round(s) with the an average perf of {score} %")
      f.close()
        
      return 
       

if __name__ == '__main__':
    parser = ArgumentParser(description="Pegasus Federated Learning Workflow Example")
    parser.add_argument('-n', type=str, help='Output file name')
    parser.add_argument('-score', type=str, help='avg score')
    parser.add_argument('-r', type=str, help='rounds executed')

    args = parser.parse_args()
    Noop().logs(args.r,args.score,args.n)
