# Here we introduce how to successfully conduct PVAED demo.
## User could work it out as follows:

step 1.   Prepare the conda environment using the environment.yml file by command: conda env create -f environment.yml (if errors come out, it's fine to omit this step and solve the question during step 4)
step 2.   Download the raw data of this demo according to data_access_info.md file.
step 3.   Run the data_process.ipynb file to get processed .h5ad data and prior results.
step 4.   Using command-line at the terminal： python main.py && main_joint_fine_tunning.py
step 5.   Run the down_stream.ipynb file to get the analysis results.
