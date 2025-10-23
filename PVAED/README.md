# Here we introduce how to successfully conduct PVAED demo.
## User could work it out as follows:

0.   Prepare the conda environment using the environment.yml file by command: conda env create -f environment.yml (if errors come out, it's fine to omit this step and solve the question in step 3)
1.   Download the raw data of this demo according to data_access_info.md file.
2.   Run the data_process.ipynb file to get processed .h5ad data and prior results.
3.   Using command-line at the terminal： python main.py && main_joint_fine_tunning.py
4.   Run the down_stream.ipynb file to get the analysis results.
