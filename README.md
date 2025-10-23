
## PVAED: Prior-Guided Variational Autoencoders with Diffusion Denoising for Interpretable Single-Cell Representation Learning



This repository contains the scripts for our PVAED model and also some jupyter notebook files showing results of the demo case. The README.md file in ./PVAED/ shows how to easily conduct our PVAED model step by step.
### PVAED model schematic:
![Workflow](PVAED/figures/F1.png)


##  Here we introduce how to successfully conduct PVAED demo.
After switch into ./PVAED folder, User could make the model work as follows:

0. Prepare the conda environment using the environment.yml file by command: conda env create -f environment.yml (if errors come out, it's fine to omit this step and solve the question in step 3)
1. Download the raw data of this demo according to data_access_info.md file.
2. Run the data_process.ipynb file to get processed .h5ad data and prior results.
3. Using command-line at the terminal： python main.py && main_joint_fine_tunning.py
4. Run the down_stream.ipynb file to get the downstream caculation results.
