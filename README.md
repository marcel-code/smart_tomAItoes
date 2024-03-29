# smart_tomAItoes
Repo for Greenhouse Challenge 2024. Smart TomAItoes for the win.

# Initialization
For using colab, use following command in one "new" notebook

```bash
!git clone https://github.com/marcel-code/smart_tomAItoes.git
```
Then, the repo will be stored at '/content/smart_tomAItoes'.

In order to make packages available, open colab

# Repo Setup
Starting code should be in the highest level (as main.py) in order to enable usability within colab and vscode.

# Setup of environment 
TBD -> conda environment?
Settings for environment to be defined (flake8, line length, ...)

Create settings file via Preferences: Open Workspace Settings (json) and add content of settings > settings_vscode.json

# Installing Anaconda
Using environment for clean distribution of dependencies - important when specific packages are needed.

Setting up environment:
```bash
conda env create -f conda_env_smart_tomAItoes.yaml  # for creating env
conda activate smart_tomAItoes  # for activating env
conda deactivate  # for deactivating env
```


