#!/usr/bin/env python3

from io import StringIO
import json
import os
import sys
import logging
import random
from pathlib import Path
from argparse import ArgumentParser
import pandas as pd

# Configure logging to display debug messages
logging.basicConfig(level=logging.DEBUG)

# --- Import Pegasus API -----------------------------------------------------------
from Pegasus.api import *

# Define a class to handle the federated learning workflow using Pegasus
class FederatedLearningWorkflow():
    # Initialize workflow-related attributes
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
        # Initialize the workflow configuration with paths and directories
        self.dagfile = dagfile
        self.wf_dir = str(Path(__file__).parent.resolve())
        self.shared_scratch_dir = os.path.join(self.wf_dir, "scratch")
        self.local_storage_dir = os.path.join(self.wf_dir, "output")
        return

    
    # --- Write files in directory -------------------------------------------------
    def write(self):
        # Write the workflow components to files if they exist
        if not self.sc is None:
            self.wf.add_site_catalog(self.sc)
            # self.sc.write() # Optional: Write the site catalog to a file
    
        self.props.write()  # Write properties file
        # self.rc.write() # Optional: Write the replica catalog to a file
        # self.tc.write() # Optional: Write the transformation catalog to a file
        self.wf.add_replica_catalog(self.rc)
        self.wf.add_transformation_catalog(self.tc)
        self.wf.write()  # Write the workflow (DAG) to a file
        return


    # --- Configuration (Pegasus Properties) ---------------------------------------
    def create_pegasus_properties(self):
        # Initialize the Pegasus properties for the workflow
        self.props = Properties()
        return


    # --- Site Catalog -------------------------------------------------------------
    def create_sites_catalog(self, exec_site_name="condorpool"):
        # Define the site catalog, specifying the local and execution sites
        self.sc = SiteCatalog()

        # Local site configuration with shared scratch and local storage directories
        local = (Site("local")
                    .add_directories(
                        Directory(Directory.SHARED_SCRATCH, self.shared_scratch_dir)
                            .add_file_servers(FileServer("file://" + self.shared_scratch_dir, Operation.ALL)),
                        Directory(Directory.LOCAL_STORAGE, self.local_storage_dir)
                            .add_file_servers(FileServer("file://" + self.local_storage_dir, Operation.ALL))
                    )
                )

        # Execution site configuration with Condor as the execution environment
        exec_site = (Site(exec_site_name)
                        .add_condor_profile(universe="vanilla")
                        .add_pegasus_profile(
                            style="condor"
                        )
                    )

        # Add the local and execution sites to the site catalog
        self.sc.add_sites(local, exec_site)
        return


    # --- Transformation Catalog (Executables and Containers) ----------------------
    def create_transformation_catalog(self, exec_site_name="condorpool"):
        # Define the transformation catalog, including executables and containers
        self.tc = TransformationCatalog()
        
        # Define the container for federated learning Using Docker
        """
        federated_learning_container = Container("federated_learning_container",
            container_type = Container.DOCKER,
            image="docker:///swarmourr/federated_learning_container_sub"
        )
        """
        # Define the container for federated learning Using SINGULARITY
        federated_learning_container = Container("federated_learning_container",
            container_type = Container.SINGULARITY,
            image=os.path.join(self.wf_dir, "../containers/fl.sif"),
            image_site="local"
        )
        
        # Define the mkdir transformation (used to create directories)
        mkdir = Transformation("mkdir", site="local", pfn="/bin/mkdir", is_stageable=False)
        
        # Define transformations for the federated learning workflow
        init_model = Transformation("init_model", site=exec_site_name, pfn=os.path.join(self.wf_dir, "../bin/init_model.py"), is_stageable=True, container=federated_learning_container).add_pegasus_profile(memory=4*1024)
        local_model = Transformation("local_model", site=exec_site_name, pfn=os.path.join(self.wf_dir, "../bin/local_model.py"), is_stageable=True, container=federated_learning_container).add_pegasus_profile(memory=4*1024)
        global_model = Transformation("global_model", site=exec_site_name, pfn=os.path.join(self.wf_dir, "../bin/global_model.py"), is_stageable=True, container=federated_learning_container).add_pegasus_profile(memory=4*1024)
        evaluate_model = Transformation("evaluate_model", site=exec_site_name, pfn=os.path.join(self.wf_dir, "../bin/evaluate_model.py"), is_stageable=True, container=federated_learning_container).add_pegasus_profile(memory=4*1024)
        perf_model = Transformation("perf_model", site=exec_site_name, pfn=os.path.join(self.wf_dir, "../bin/perf_model.py"), is_stageable=True, container=federated_learning_container).add_pegasus_profile(memory=4*1024)
        build_local_model = Transformation("build_local_workflow", site=exec_site_name, pfn=os.path.join(self.wf_dir, "workflow_generator_sub.py"), is_stageable=True, container=federated_learning_container).add_pegasus_profile(memory=4*1024)
    
        # Add the container and transformations to the catalog
        self.tc.add_containers(federated_learning_container)
        self.tc.add_transformations(init_model, local_model, global_model, evaluate_model, perf_model, build_local_model)
        return


    # --- Replica Catalog ----------------------------------------------------------
    def create_replica_catalog(self, model_path=""):
        # Define the replica catalog, mapping logical filenames to physical locations
        self.rc = ReplicaCatalog()
        self.rc.add_replica("local", "train-images-idx3-ubyte", os.path.join(self.wf_dir, "../mnist", "train-images-idx3-ubyte"))
        self.rc.add_replica("local", "train-labels-idx1-ubyte", os.path.join(self.wf_dir, "../mnist", "train-labels-idx1-ubyte"))
        self.rc.add_replica("local", "t10k-images-idx3-ubyte", os.path.join(self.wf_dir, "../mnist", "t10k-images-idx3-ubyte"))
        self.rc.add_replica("local", "t10k-labels-idx1-ubyte", os.path.join(self.wf_dir, "../mnist", "t10k-labels-idx1-ubyte"))
        if model_path != "":
            self.rc.add_replica("local", "global_model_round_init.h5", os.path.join(self.wf_dir, "output", model_path))
        return


    # --- Create Workflow ----------------------------------------------------------
    def create_workflow(self, clients, selection, rounds, model_path, initiation, round, score):
        # Create the federated learning workflow, specifying tasks and dependencies
        self.wf = Workflow(self.wf_name, infer_dependencies=True)
        
        mnist_train_size = 60000  # Fixed training dataset size
        mnist_test_size = 10000  # Fixed test dataset size
        
        # Distribute data among clients
        images_per_client = mnist_train_size // clients
        images_test_per_client = mnist_test_size // clients
        client_list = [i for i in range(clients)]
        eval_test = 80  # Evaluation test percentage
        train_images = File("train-images-idx3-ubyte")
        train_labels = File("train-labels-idx1-ubyte")
        test_images = File("t10k-images-idx3-ubyte")
        test_labels = File("t10k-labels-idx1-ubyte")
        global_model_name = ""
        local_model_base_name = "local_wf_job"
        global_model_base_name = "global_model_job"
        
        if initiation:
            # Step 1: Build initial global model
            model_path = "global_model_round_init.h5"
            global_model = File(f"global_model_round_init.h5")
            initial_model = (Job("init_model", _id="init_model", node_label=f"init_model")
                             .add_args(f"-n global_model_round_init.h5")
                             .add_outputs(global_model, stage_out=True, register_replica=False)        
            )    
            self.wf.add_jobs(initial_model)
    
        # Define local workflow jobs for each round
        wf_outputs = []
        local_wf_output = File(f"local_wf_round_{round}.yml")
        wf_outputs.append(local_wf_output)
        locals()[local_model_base_name + str(round)] = (Job(f"build_local_workflow", _id=f"round_{round}", node_label=f"build_local_workflow_round_{round}")\
                .add_args(f"-c {clients} -n {selection} -r {round} --output local_wf_round_{round}.yml -td {train_images} -tl {train_labels} -tsd {test_images} -tsl {test_labels} -m {model_path} -score {score} -max {rounds} -cr {round}" )\
                .add_inputs(train_images, train_labels, test_images, test_labels, global_model)\
                .add_outputs(local_wf_output, stage_out=True, register_replica=False)          
            )
        self.wf.add_jobs(locals()[local_model_base_name + str(round)])
        
        # (Optional) Uncomment the following block to execute sub-workflows for each round
        # """result_file_name=f"Model_performances_{round}.csv"
        # Final_model_Results = File(result_file_name)
        # locals()["local_subwf_" + str(round)]  = SubWorkflow(local_wf_output, False, _id=f"local_wf_round_{round}")\
        #     .add_args("--sites", "condorpool",
        #               "--basename", f"local_wf_round_{round}",
        #               "--force",
        #               "--output-site", "local")\
        #     .add_inputs(global_model, train_images, train_labels, test_images, test_labels, local_rc_yml)\
        #     .set_stdout(Final_model_Results, stage_out=True, register_replica=True)
        # self.wf.add_jobs(locals()["local_subwf_" + str(round)])"""
                             
        return 
        
    # --- Run Workflow ----------------------------------------------------------
    def run_workflow(self, execution_site_name, skip_sites_catalog, clients, number_of_selected_clients, number_of_rounds, initiation, model_path, round, score):
        # Validate the number of clients
        if clients < 1:
            print("Clients number needs to be > 0")
            exit()
        
        # Create the site catalog if not skipped
        if not skip_sites_catalog:
            self.create_sites_catalog(execution_site_name)
    
        print("Creating workflow properties...")
        self.create_pegasus_properties()
        
        print("Creating transformation catalog...")
        self.create_transformation_catalog(execution_site_name)
    
        print("Creating replica catalog...")
        self.create_replica_catalog(model_path)
    
        print("Creating the federated learning workflow dag...")
        self.create_workflow(clients, number_of_selected_clients, number_of_rounds, model_path, initiation, round, score)
       
        self.write()  # Write the workflow components to files
        
    

if __name__ == '__main__':
    # Parse command-line arguments
    parser = ArgumentParser(description="Pegasus Federated Learning Workflow Example")

    # Define the command-line arguments
    parser.add_argument("-s", "--skip-sites-catalog", action="store_true", help="Skip site catalog creation")
    parser.add_argument("-e", "--execution-site-name", metavar="STR", type=str, default="condorpool", help="Execution site name (default: condorpool)")
    parser.add_argument("-o", "--output", metavar="STR", type=str, default="workflow.yml", help="Output file (default: workflow.yml)")
    parser.add_argument("-c", "--clients", metavar="INT", type=int, default=1, help="Number of available clients (default: 1)")
    parser.add_argument("-n", "--number-of-selected-clients", metavar="INT", type=int, default=1, help="Number of selected clients (default: 1)")
    parser.add_argument("-r", "--number-of-rounds", metavar="INT", type=int, default=1, help="Number of rounds (default: 1)")
    parser.add_argument("-score", metavar="INT", type=int, default=100, help="Score to stop the training")
    parser.add_argument("-cr", metavar="INT", type=int, default=0, help="Current round")
    args = parser.parse_args()
        
    # Validate the number of clients
    if args.clients < 1:
        print("Clients number needs to be > 0")
        exit()
    
    # Initialize the workflow
    workflow = FederatedLearningWorkflow(dagfile=args.output)
    round = args.cr
    print(f"Current round: {round}")
    if int(round) == 0: 
        print("Starting initial round")
        model_path = ""
        initiation = True
    else:
        initiation = False
        model_path = f"global_model_round_{int(round)-1}.h5"
    
    # Run the federated learning workflow
    workflow.run_workflow(args.execution_site_name, args.skip_sites_catalog, args.clients, args.number_of_selected_clients, args.number_of_rounds, initiation, model_path, round, args.score)
    
    try:
        # Plan and submit the workflow to Pegasus
        workflow.wf.plan(submit=True,sites=["condorpool"], output_sites=["local"]).wait()
    except PegasusClientError as e:
        print(e.output)
