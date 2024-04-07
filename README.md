# smart_tomAItoes
Repo for Greenhouse Challenge 2024. Smart TomAItoes for the win.

# Initialization
For using colab, use following command in one "new" notebook

```bash
!git clone https://github.com/marcel-code/smart_tomAItoes.git
```
Then, the repo will be stored at '/content/smart_tomAItoes'.

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

# Including dataset into local environment
Add image dataset into local data folder

## Usage of pipeline
# Adding new features to config file
For adding new setting values to the config file, make sure to add the according value to the dataloader (TomatoDataset - default_config)

# Adding new model
For adding a new model for the evaluation, add a new .py file including the new model. Make sure that the model inherits from torch.nn.Module Class, in order to be aligned with the dynamic model loader (get_model). Try to keep aligned with DummyModel regarding the structure.

# Adding losses
Adding custom loss to the src.models.utils.losses.py file and load it into your model for the loss_fn

# Execute Training
python train.py <Experimentname> --conf <path to config>