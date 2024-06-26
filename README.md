# smart_tomAItoes
Repo for Greenhouse Challenge 2024. Smart TomAItoes for the win.

# Initialization
For using colab, use the Notebook within colab.
The first part will initialize everything needed
  - cloning of current main version of code
  - including repo location to system path
  - checking if setup was successfull (simple test)
  -
The repo will be stored at '/content/smart_tomAItoes'.

# Repo Setup
Starting code should be in the highest level (as main.py) in order to enable usability within colab and vscode.

## Setup of environment 
# Extensions to install
- flake8 - ms-python.flake8 -> similar layout (settings in workspace important)
- black formatter - ms-python.black-formatter -> similar layout (settings in workspace important)
- isort - ms-python.isort -> sorting imported modules
- autodocstring - njpwerner.autodocstring -> good for pre-defined docstring
- todo Tree - gruntfuggly.todo-tree -> overview of TODO (case-sensitive) Comments within project

# Conda Environment
TODO requirements.txt to install dependencies on colab necessary

Create settings file via Preferences: Open Workspace Settings (json) and add content of settings > settings_vscode.json

# Installing Anaconda
Using environment for clean distribution of dependencies - important when specific packages are needed.

Setting up environment:
```bash
conda env create -f conda_env_smart_tomAItoes.yaml  # for creating env
conda activate smarties_partA  # for activating env
conda deactivate  # for deactivating env
```

Export conda environment:
```bash
conda env export > filename.yaml
```

# Including dataset into local environment
Add image dataset into local data folder

# Pointcloud Generation for depth image
Submodule point-cloud-demo from university wagening as thrid party provider included. Takes two images and camera instrinsics in order to calculate point cloud. Look at the demo for more details.

## Usage of pipeline
# Adding new features to config file
For adding new setting values to the config file, make sure to add the according value to the dataloader (TomatoDataset - default_config)

# Adding new model
For adding a new model for the evaluation, add a new .py file including the new model. Make sure that the model inherits from torch.nn.Module Class, in order to be aligned with the dynamic model loader (get_model). Try to keep aligned with DummyModel regarding the structure.

For preventing certain layers from training, just set the requires_grad argument to False.

# Adding losses
Adding custom loss to the src.models.utils.losses.py file. Dynamic loading by changing name in config file. Make sure to implement a Class for your specific loss.

# Adding Optimizer
For using another optimizer than SGD, you have to add the according arguments to the .yaml file - same style as already with lr and momentum for SGD. Otherwise, the dynamic selection will fail.

# Execute Training
python train.py <Experimentname> --conf <path to config>

# Tensorboard
During or after training, you can open tensorboard to visualize the progress and changes in losses, learning rates and similar (defined in train.py with writer.add_scalar or similar).

To activate tensorboard, run 
```bash
tensorboard --logdir=outputs --host=localhost --port=6007
```

I didn't get it working for opening the browser directly. However, you can either click on the local host link in your terminal (http://localhost:6007/) or copying the weblink directly into your preferred browser.

## Colab
# Setup
Select repo in colab under github in order to use the according files. In order to clone the correct branch, you have to adapt the clone command in the notebook.ipynb (just change main to whatever your branch is named)
Happy working.