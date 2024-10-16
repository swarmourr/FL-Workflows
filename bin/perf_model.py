#!/usr/bin/env python3

import pandas as pd 
import numpy as np 
from argparse import ArgumentParser

import os
import sys
import logging
import random
from pathlib import Path
from argparse import ArgumentParser
import pandas as pd

#logging.basicConfig(level=logging.DEBUG)
#str(Path(__file__).parent.resolve())

# --- Import Pegasus API -----------------------------------------------------------
from Pegasus.api import *


class FederatedLearningWorkflow():
    wf = None
    sc = None
    tc = None
    rc = None
    props = None

    dagfile = None
    wf_dir = None
    shared_scratch_dir = None
    local_storage_dir = None
    wf_name = "federated-learning-example"
    
    # --- Init ---------------------------------------------------------------------
    def __init__(self, dagfile="workflow.yml"):
        self.dagfile = dagfile
        self.wf_dir = "/home/hsafri/FederatedLearning/federated-learning-score-EM-blog/Main_workflow/" #str(Path(__file__).parent.resolve())
        self.shared_scratch_dir = "/home/hsafri/FederatedLearning/federated-learning-score-EM-blog/Main_workflow/scratch/" #os.path.join(self.wf_dir, "Main_workflow/scratch")
        self.local_storage_dir = "/home/hsafri/FederatedLearning/federated-learning-score-EM-blog/Main_workflow/output/" #os.path.join(self.wf_dir, "Main_workflow/output")
        return

    
    # --- Write files in directory -------------------------------------------------
    def write(self,name):
        if not self.sc is None:
            self.wf.add_site_catalog(self.sc)
    
        self.props.write()
        #self.rc.write()
        #self.tc.write()
        self.wf.add_replica_catalog(self.rc)
        self.wf.add_transformation_catalog(self.tc)
        self.wf.write(name)
        return



    # --- Configuration (Pegasus Properties) ---------------------------------------
    def create_pegasus_properties(self):
        self.props = Properties()
        return


    # --- Site Catalog -------------------------------------------------------------
    def create_sites_catalog(self, exec_site_name="condorpool"):
        self.sc = SiteCatalog()
        
        local = (Site("local")
                    .add_directories(
                        Directory(Directory.SHARED_SCRATCH, self.shared_scratch_dir)
                            .add_file_servers(FileServer("file://" + self.shared_scratch_dir, Operation.ALL)),
                            
                        Directory(Directory.LOCAL_STORAGE, self.local_storage_dir)
                            .add_file_servers(FileServer("file://" + self.local_storage_dir, Operation.ALL))
                    )
                )

        exec_site = (Site(exec_site_name)
                        .add_condor_profile(universe="vanilla")
                        .add_pegasus_profile(
                            style="condor"
                        )
                    )

        
        self.sc.add_sites(local, exec_site)
        return


    # --- Transformation Catalog (Executables and Containers) ----------------------
    def create_transformation_catalog(self, exec_site_name="condorpool"):
        self.tc = TransformationCatalog()
        
        federated_learning_container = Container("federated_learning_container",
            container_type = Container.SINGULARITY,
            image=os.path.join(self.wf_dir, "../containers/fl.sif"),
            image_site="local"
        )
        
        # Add the orcasound processing
        mkdir = Transformation("mkdir", site="local", pfn="/bin/mkdir", is_stageable=False)
        
        
        init_model = Transformation("init_model", site=exec_site_name, pfn=os.path.join(self.wf_dir, "../bin/init_model.py"), is_stageable=True, container=federated_learning_container).add_pegasus_profile(memory=4*1024)
        local_model = Transformation("local_model", site=exec_site_name, pfn=os.path.join(self.wf_dir, "../bin/local_model.py"), is_stageable=True, container=federated_learning_container).add_pegasus_profile(memory=4*1024)
        global_model = Transformation("global_model", site=exec_site_name, pfn=os.path.join(self.wf_dir, "../bin/global_model.py"), is_stageable=True, container=federated_learning_container).add_pegasus_profile(memory=4*1024)
        evaluate_model = Transformation("evaluate_model", site=exec_site_name, pfn=os.path.join(self.wf_dir, "../bin/evaluate_model.py"), is_stageable=True, container=federated_learning_container).add_pegasus_profile(memory=4*1024)
        perf_model = Transformation("perf_model", site=exec_site_name, pfn=os.path.join(self.wf_dir, "../bin/perf_model.py"), is_stageable=True, container=federated_learning_container).add_pegasus_profile(memory=4*1024)
        build_local_model = Transformation("build_local_workflow", site=exec_site_name, pfn=os.path.join(self.wf_dir, "workflow_generator_sub.py"), is_stageable=True, container=federated_learning_container).add_pegasus_profile(memory=4*1024)
    
        
        self.tc.add_containers(federated_learning_container)
        self.tc.add_transformations(init_model, local_model, global_model, evaluate_model, perf_model, build_local_model)
        return


    # --- Replica Catalog ----------------------------------------------------------
    def create_replica_catalog(self, model_path=""):
        self.rc = ReplicaCatalog()
        
        self.rc.add_replica("local", "train-images-idx3-ubyte", "/home/hsafri/FederatedLearning/federated-learning-score-EM-blog/mnist/train-images-idx3-ubyte")
        self.rc.add_replica("local", "train-labels-idx1-ubyte", "/home/hsafri/FederatedLearning/federated-learning-score-EM-blog/mnist/train-labels-idx1-ubyte")
        self.rc.add_replica("local", "t10k-images-idx3-ubyte", "/home/hsafri/FederatedLearning/federated-learning-score-EM-blog/mnist/t10k-images-idx3-ubyte")
        self.rc.add_replica("local", "t10k-labels-idx1-ubyte", "/home/hsafri/FederatedLearning/federated-learning-score-EM-blog/mnist/t10k-labels-idx1-ubyte")
        
        if model_path != "":
            self.rc.add_replica("local", model_path, "/home/hsafri/FederatedLearning/federated-learning-score-EM-blog/Main_workflow/output/"+model_path)
        return


    # --- Create Workflow ----------------------------------------------------------
    def create_workflow(self, clients, selection, rounds, model_path,initiation, main_round,score):
        self.wf = Workflow(self.wf_name, infer_dependencies=True)
        
        
        mnist_train_size = 60000 #these are fixed limits based on the dataset
        mnist_test_size = 10000 #these are fixed limits based on the dataset
        
        #for now let's give each client sequential image range --> [client*images_per_client:(client+1)*images_per_client-1]
        #for the last client let's give [client*images_per_client:mnist_training_size]
        images_per_client = mnist_train_size // clients
        images_test_per_client=mnist_test_size // clients
        client_list = [i for i in range(clients)]
        eval_test=80
        train_images = File("train-images-idx3-ubyte")
        train_labels = File("train-labels-idx1-ubyte")
        test_images = File("t10k-images-idx3-ubyte")
        test_labels = File("t10k-labels-idx1-ubyte")
        global_model_name=""
        
        if initiation:
          #step 1 : Build intial model 
          global_model = File(f"global_model_round_init.h5")
          initial_model= (Job("init_model", _id=f"init_model", node_label=f"init_model")
                  .add_args(f"-n global_model_round_init.h5")
                  .add_outputs(global_model, stage_out=True, register_replica=False)        
          )
                 
          #self.wf.add_jobs(initial_model)
         
        # step by rounds 
        #for round in range(1):  
        #step 1 DONE: read the client list
        #step 2 DONE: split data into buckets
                        
        """if initiation ==True :
        print(f"initiate the model for {main_round}")
        global_model_name="global_model_round_init.h5"
        else:
        print(f"output/global_model_round_{main_round-1}.h5")
        global_model_name=model_path"""
        
        global_model = File(f"global_model_round_{main_round-1}.h5")
            
        #step 3 DONE: select clients for FL
        selected_clients = random.sample(client_list, k=selection)

        #step 4 TODO: BUILD the initial global model and save in ./models
        #step 4 TODO: ADJUST the local_model.py to accept the ranges and save in ./bin
        #step 4: foreach client build local model(s)
        local_model_base_name="local_model_job"
        global_model_base_name="global_model_job"
        #step 4a: preprocess data
        #step 4b: build local model(s)
        local_model_outputs = []
        for client in selected_clients:
            start_position = client*images_per_client
            end_position = (client+1)*images_per_client - 1
            local_model_output = File(f"local_model_{start_position}_{end_position}_round_{main_round}.h5")
            local_model_outputs.append(local_model_output)
            locals()[local_model_base_name+str(main_round)] = (Job(f"local_model", _id=f"client_{client}_round_{main_round}", node_label=f"local_model_client_{client}_round_{main_round}")
                .add_args(f"-s {start_position} -e {end_position} -r {main_round} -m global_model_round_{main_round-1}.h5")
                .add_inputs(train_images, train_labels, global_model)
                .add_outputs(local_model_output, stage_out=True, register_replica=False)
            )
            
            self.wf.add_jobs(locals()[local_model_base_name+str(main_round)] )
            
        #step 5 DONE: build global model
        global_model = File(f"global_model_round_{main_round}.h5")
        locals()[global_model_base_name+str(main_round)]  = (Job("global_model", _id=f"global_model_round_{main_round}", node_label=f"global_model_round_{main_round}")
            .add_args(f"-r {main_round} -f {' '.join([x.lfn for x in local_model_outputs])}")
            .add_inputs(*local_model_outputs)
            .add_outputs(global_model, stage_out=True, register_replica=False)
        )
            
        self.wf.add_jobs(locals()[global_model_base_name+str(main_round)] )

        last_model_name=f"global_model_round_{main_round}.h5"
        print("hadi last  name model : " + last_model_name)


                    
        #step 6 DONE: select clients for evaluation
        selected_eval_clients = random.sample(client_list, k=selection)
                
        #step 7 TODO: edit evalute script to accept client number and name the output based on that
        #step 7 TODO: read the mnist test dataset from the single files
        #step 7 TODO: add a preprocess part in the evaluation script
        #step 7 DONE: foreach client run evaluation
                
        evaluate_model_outputs = []
        for client in selected_eval_clients:
            start_position = client*images_test_per_client
            end_position = (client+1)*images_test_per_client - 1
            evaluate_model_output = File(f"global_model_evaluation_{client}_{main_round}.csv")
            evaluate_model_outputs.append(evaluate_model_output)
            evaluate_model_job = (Job("evaluate_model", _id=f"eval_client_{client}_{main_round}", node_label=f"evaluate_model_client_{client}_{main_round}")
                .add_args(f"-s {start_position} -e {end_position} -c {client} -m {last_model_name} -r {main_round}")
                .add_inputs(global_model, test_images, test_labels)
                .add_outputs(evaluate_model_output, stage_out=True, register_replica=False)
            )
                        
            self.wf.add_jobs(evaluate_model_job)
            
                        
        #step 8: get global evaluation score
        result_file_name=f"Model_performences_{main_round}.csv"
        Final_model_Results = File(result_file_name)
        #replicas_files=File("replicas.yml")
        new_sub_wf=File(f"local_wf_round_{int(main_round)+1}.yml")
        model_perf_job = (Job("perf_model", _id=f"perf_model_{main_round}", node_label=f"perf_model_format_{main_round}")
            .add_args(f"-f {' '.join([x.lfn for x in evaluate_model_outputs])}  -n {result_file_name} -r {main_round} -score {score} -max {rounds}")
            .add_inputs(*evaluate_model_outputs)   
            .add_outputs(Final_model_Results,new_sub_wf, stage_out=True, register_replica=False)
        ) 

        self.wf.add_jobs(model_perf_job)

        """locals()["local_subwf_"+str(main_round+1)]  = SubWorkflow(new_sub_wf, False, _id=f"local_wf_round_{main_round+1}")\
            .add_args("--sites", "condorpool",
                        "--basename", f"local_wf_round_{main_round+1}",
                        "--force",
                        "--output-site", "local")\
            .add_inputs(File(last_model_name), train_images, train_labels, test_images, test_labels,replicas_files)\
            #.set_stdout(Final_model_Results, stage_out=True, register_replica=True)

        self.wf.add_jobs(locals()["local_subwf_"+str(main_round+1)])"""
                        
        
        return self.wf 
        
    # --- Run Workflow ----------------------------------------------------------
    def run_workflow(self,execution_site_name, skip_sites_catalog,clients, number_of_selected_clients, number_of_rounds,initiation, model_path, round,name,score):
        if clients < 1:
            print("Clients number needs to be > 0")
            exit()
        
        if not skip_sites_catalog:
            print("Creating execution sites...")
            self.create_sites_catalog(execution_site_name)
        print("Creating execution sites...")
        self.create_sites_catalog(execution_site_name)

        print("Creating workflow properties...")
        self.create_pegasus_properties()
        
        print("Creating transformation catalog...")
        self.create_transformation_catalog(execution_site_name)
    
        print("Creating replica catalog...")
        self.create_replica_catalog(model_path)
    
        print("Creating the federated learning workflow dag...")
        self.create_workflow(clients, number_of_selected_clients, number_of_rounds,model_path,initiation,round,score)
       
        self.write(name)



class NoopWorkflow():
    wf = None
    sc = None
    tc = None
    rc = None
    props = None

    dagfile = None
    wf_dir = None
    shared_scratch_dir = None
    local_storage_dir = None
    wf_name = "federated-learning-example"
    
    # --- Init ---------------------------------------------------------------------
    def __init__(self, dagfile="workflow.yml"):
        self.dagfile = dagfile
        self.wf_dir = "/home/hsafri/FederatedLearning/federated-learning-score-EM-blog/Main_workflow/" #str(Path(__file__).parent.resolve())
        self.shared_scratch_dir = "/home/hsafri/FederatedLearning/federated-learning-score-EM-blog/Main_workflow/scratch" #os.path.join(self.wf_dir, "Main_workflow/scratch")
        self.local_storage_dir = "/home/hsafri/FederatedLearning/federated-learning-score-EM-blog/Main_workflow/output" #os.path.join(self.wf_dir, "Main_workflow/output")
        return

    
    # --- Write files in directory -------------------------------------------------
    def write(self,name):
        if not self.sc is None:
            self.wf.add_site_catalog(self.sc)
    
        self.props.write()
        #self.rc.write()
        #self.tc.write()
        self.wf.add_replica_catalog(self.rc)
        self.wf.add_transformation_catalog(self.tc)
        self.wf.write(name)
        return self.wf 



    # --- Configuration (Pegasus Properties) ---------------------------------------
    def create_pegasus_properties(self):
        self.props = Properties()
        return


    # --- Site Catalog -------------------------------------------------------------
    def create_sites_catalog(self, exec_site_name="condorpool"):
        self.sc = SiteCatalog()
        
        local = (Site("local")
                    .add_directories(
                        Directory(Directory.SHARED_SCRATCH, self.shared_scratch_dir)
                            .add_file_servers(FileServer("file://" + self.shared_scratch_dir, Operation.ALL)),
                            
                        Directory(Directory.LOCAL_STORAGE, self.local_storage_dir)
                            .add_file_servers(FileServer("file://" + self.local_storage_dir, Operation.ALL))
                    )
                )

        exec_site = (Site(exec_site_name)
                        .add_condor_profile(universe="vanilla")
                        .add_pegasus_profile(
                            style="condor"
                        )
                    )

        
        self.sc.add_sites(local, exec_site)
        return


    # --- Transformation Catalog (Executables and Containers) ----------------------
    def create_transformation_catalog(self, exec_site_name="condorpool"):
        self.tc = TransformationCatalog()
        
        federated_learning_container = Container("federated_learning_container",
            container_type = Container.SINGULARITY,
            image=os.path.join(self.wf_dir, "../containers/fl.sif"),
            image_site="local"
        )
        
        # Add the orcasound processing
        mkdir = Transformation("mkdir", site="local", pfn="/bin/mkdir", is_stageable=False)
        
        
        
        noop = Transformation("noop", site=exec_site_name, pfn=os.path.join(self.wf_dir, "../bin/noop.py"), is_stageable=True, container=federated_learning_container).add_pegasus_profile(memory=4*1024)
    
        
        self.tc.add_containers(federated_learning_container)
        self.tc.add_transformations(noop)
        return


    # --- Replica Catalog ----------------------------------------------------------
    def create_replica_catalog(self, model_path=""):
        self.rc = ReplicaCatalog()
        
      
        return


    # --- Create Workflow ----------------------------------------------------------
    def create_workflow(self,main_round,score):
        self.wf = Workflow(self.wf_name, infer_dependencies=True)
        
       
        #step 1 : Build intial model 
        noop = File(f"final_log.txt")
        noop_job= (Job("noop", _id=f"noop", node_label=f"noop")
                  .add_args(f"-n final_log.txt -r {main_round} -score {score}")
                  .add_outputs(noop, stage_out=True, register_replica=False)        
          )        
        self.wf.add_jobs(noop_job)
         
        
        return self.wf 
        
    # --- Run Workflow ----------------------------------------------------------
    def run_workflow(self,execution_site_name, skip_sites_catalog,round,name,score):
        
        
        if not skip_sites_catalog:
            print("Creating execution sites...")
            self.create_sites_catalog(execution_site_name)

        print("Creating execution sites...")
        self.create_sites_catalog(execution_site_name)
        
        print("Creating workflow properties...")
        self.create_pegasus_properties()
        
        print("Creating transformation catalog...")
        self.create_transformation_catalog(execution_site_name)
    
        print("Creating replica catalog...")
        self.create_replica_catalog()
    
        print("Creating the federated learning workflow dag...")
        self.create_workflow(round,score)
        
        self.write(name)

class ModelPerf():
    
    def __init__(self):
        pass
        
    def concatResults(self,paths,name,round,max,score):
       perfermence_df = pd.DataFrame()
       for file in paths:
          data = pd.read_csv(file)
          perfermence_df = pd.concat([perfermence_df, data], axis=0)
       perfermence_df['avg_acc']=perfermence_df.acc.mean()
       perfermence_df['avg_loss']=perfermence_df.loss.mean()
       perfermence_df.to_csv(name, index=False)
       acc=perfermence_df['avg_acc'].values[0]
       wf_name=f"local_wf_round_{int(round)+1}.yml"
       mdl_name=f"global_model_round_{round}.h5"
       rplc_tmp="replicas.yml"
       if acc > int(score) or int(round) == (int(max)-1) :
            sub_wf=NoopWorkflow(wf_name)
            sub_wf.run_workflow("condorpool",True,int(round)+1,wf_name,acc)
            print("Data has been written to the file as YAML successfully.")
       else : 
          sub_wf=FederatedLearningWorkflow(wf_name)
          sub_wf.run_workflow("condorpool",True,10, 2,max,False,mdl_name,int(round)+1,wf_name,score)
          print("run new round")
         
       return 
        

if __name__ == '__main__':
    parser = ArgumentParser(description="Pegasus Federated Learning Workflow Example")
    parser.add_argument('-f', type=str, nargs='+', help='Path to local models')
    parser.add_argument('-n', type=str, help='Output file name', default=None)
    parser.add_argument('-r', type=str, help='current round name', default=None)
    parser.add_argument('-max', type=str, help='Max rounds', default=None)
    parser.add_argument('-score', type=int, help='Model perf score', default=100)
    args = parser.parse_args()
    ModelPerf().concatResults(args.f,args.n,args.r,args.max,args.score)
