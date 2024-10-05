![Static Badge](https://custom-icon-badges.demolab.com/badge/pegasus-5.0.1-wms?style=flat&logo=pegasus-wms&color=blue&link=https%3A%2F%2Fgithub.com%2Fpegasus-isi%2Fpegasus)
![Static Badge](https://custom-icon-badges.demolab.com/badge/apptainer-1.3.3--1.el8-wms?style=flat&logo=apptainer&color=blue&link=https%3A%2F%2Fgithub.com%2Fpegasus-isi%2Fpegasus)
![PyPI - Python Version](https://img.shields.io/pypi/pyversions/pegasus-wms?link=https%3A%2F%2Fgithub.com%2Fpegasus-isi%2Fpegasus)

# Federated Learning Workflow Management

This `README.md` provides an overview of the federated learning workflow management setup using Pegasus WMS. It includes details on script usage, how to modify paths, and setup triggers.

## Directory Structure

```bash
federated-learning-score-EM-blog
├── bin
│   ├── evaluate_model.py
│   ├── global_model.py
│   ├── init_model.py
│   ├── __init__.py
│   ├── local_model.py
│   ├── noop.py
│   ├── perf_model.py
│   └── perf_model_copy.py
├── containers
│   └── fl.sif
├── __init__.py
├── Main_workflow
│   ├── __init__.py
│   ├── pegasus.properties
│   ├── plan.sh
│   ├── trigger-Copy1.sh
│   ├── trigger.sh
│   ├── workflow_generator_for_loop.py
│   ├── workflow_generator_main_sub.py
│   └── workflow_generator_sub.py
├── mnist
│   ├── t10k-images-idx3-ubyte
│   ├── t10k-images-idx3-ubyte.gz
│   ├── t10k-labels-idx1-ubyte
│   ├── t10k-labels-idx1-ubyte.gz
│   ├── train-images-idx3-ubyte
│   ├── train-images-idx3-ubyte.gz
│   ├── train-labels-idx1-ubyte
│   └── train-labels-idx1-ubyte.gz
├── models
├── pegasus.properties
└── README.md
```

## Scripts Overview

### 1. `workflow_generator_main_sub.py`

This script generates the main workflow for federated learning. It handles:

- **Site and Replica Catalogs**: Defines the execution sites and data replicas.
- **Transformations**: Specifies the scripts used in the workflow.
- **Workflow YAML**: Creates the YAML file required for running the workflow.

#### Paths to Modify

- **Workflow Directory (`self.wf_dir`)**:
  - **Default**: `/home/hsafri/FederatedLearning/federated-learning-score-EM-blog/Main_workflow/`
  - **Modification**: Update to your directory where `workflow_generator_main_sub.py` is located.

- **Shared Scratch Directory (`self.shared_scratch_dir`)**:
  - **Default**: `/home/hsafri/FederatedLearning/federated-learning-score-EM-blog/Main_workflow/scratch/`
  - **Modification**: Set this to your environment’s shared scratch directory.

- **Local Storage Directory (`self.local_storage_dir`)**:
  - **Default**: `/home/hsafri/FederatedLearning/federated-learning-score-EM-blog/Main_workflow/output/`
  - **Modification**: Update this to your output directory.

- **Transformation Scripts Paths**:
  - **Default**:
    - `init_model.py`: `../bin/init_model.py`
    - `local_model.py`: `../bin/local_model.py`
    - `global_model.py`: `../bin/global_model.py`
    - `evaluate_model.py`: `../bin/evaluate_model.py`
    - `perf_model.py`: `../bin/perf_model.py`
    - `workflow_generator_sub.py`: `workflow_generator_sub.py`
  - **Modification**: Adjust paths according to the actual locations of these scripts.

- **Replica Catalog Paths**:
  - Ensure paths to MNIST dataset files are accurate:
    - Example: `"/home/hsafri/FederatedLearning/federated-learning-score-EM-blog/mnist/train-images-idx3-ubyte"`
  - **Modification**: Change these to your data files' locations.

### 2. `workflow_generator_sub.py`

Generates and manages workflows for first federated learning round.

### 3. `workflow_generator_for_loop.py`

Creates multiple Federated Learning rounds in a loop if needed (Other implementation Example).

## Trigger Setup

Triggers are used to automatically submit workflows based on file patterns using Pegasus Ensemble Manager (EM).

### Trigger Script: `trigger.sh`

1. **Create Ensemble**

   ```bash
   echo Creating FL_ensemble if not exist
   pegasus-em create FL_ensemble2 2>/dev/null
   ```

   - **Modification**: Ensure `FL_ensemble2` is the correct ensemble name or update it as needed.

2. **Create Trigger**

   ```bash
   echo Creating triggers if not exist
   pegasus-em file-pattern-trigger "FL_ensemble2" "local_training_$f2_$(echo $RANDOM | md5sum | head -c 20)" 10s "/path/to/plan.sh" "/path/to/output/local_wf_round_*.yml" --timeout 90m
   ```

   - **Ensemble Name**: `FL_ensemble2`
   - **Trigger Name**: `local_training_$f2_$(echo $RANDOM | md5sum | head -c 20)`
   - **Trigger Interval**: `10s`
   - **Plan Path**: `/path/to/plan.sh` (Update to the actual path of your Pegasus plan script)
   - **Output Pattern**: `/path/to/output/local_wf_round_*.yml` (Update to match your workflow YAML files' location and pattern)
   - **Timeout**: `90m`

   **Modification**: Ensure all paths in `file-pattern-trigger` are correct.

## Running the Scripts

1. **Generate and Run Main Workflow**:

   ```bash
   python3 Main_workflow/workflow_generator_main_sub.py -c <clients> -n <selection> -r <rounds> -score <score> -cr <current_round>
   ```

   - **`-c`**: Number of clients.
   - **`-n`**: Number of selected clients.
   - **`-r`**: Number of rounds.
   - **`-score`**: Maximum score percentage to achieve before stopping the training.
   - **`-cr`**: Current round number or relevant parameter.

2. **Set Up Triggers**:

   ```bash
   bash Main_workflow/trigger.sh
   ```

## Conclusion

Ensure that all paths are updated according to your environment before running the scripts. This setup will help you manage federated learning workflows effectively with Pegasus WMS.

For further configurations or troubleshooting, refer to the [Pegasus documentation](https://pegasus.isi.edu/documentation/) or seek additional support pegasus-support@isi.edu.

