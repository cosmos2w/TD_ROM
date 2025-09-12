# This document describes the minimum standard we follow for
- ingesting heterogeneous scientific datasets,
- training different reduced-order modelling (ROM) approaches, and
- reporting results in a fully reproducible way.

## The model will take the information of selected observation data in a time window (N_obs) as input, and perform: 
    (1) full-field reconstruction for sparse observations
    (2) forecasting the evolution of spatiotemporal dynamics in N_fore more time steps.

# 1.Directory Layout

├── dataset/                    # Raw + processed data (never committed to git); One sub-directory per physical system
│   ├── raw/                    # Original files exactly as downloaded/generated
│   ├── processed/              # HDF5 files that follow the template below
│   └── README.md               # Origin & case description
├── save_config_files/          # Frozen YAML configuration files
├── save_check_files/           # Save the intermediate feature checking files
├── save_loss_files/            # CSV loss logs + loss history figures
├── Output_Net/                 # Best checkpoints (*.pth)
├── Save_reconstruction_files/  # ROM reconstructions, latent states, etc.
├── src/                        # Importable Python package
│   ├── convert.py              # Converts `raw/` → `processed/`
│   ├── train_{...}.py          # Single-GPU/CPU training entry point
│   ├── evaluate_{...}.py       # Inference + metric calculation
│   ├── dataloading.py
│   ├── models.py
│   ├── metrics.py
│   ├── utils/
│   │   ├── io.py
│   │   └── plot.py
│   └── baselines/              # POD, FNO, AE-CNN, …
└── README.md                   # ← you are here


- One Python entry point per stage (convert.py, train.py, evaluate.py).
- Everything that depends on hyper-parameters must be controlled exclusively through a YAML config file.

# 2. Data preprocessing, reshaping and organization

## For a given problem (Burgers equation, NS equation, etc.), the dataset should contain the spatiotemporal evolution under different circumstances (varying parameters or initial conditions).

All data should be reformatted into HDF5 files and organized in a dataset: with separated field, spatial coordinate and time information. 
For instance, for a 3-D dataset with 2 fields, it should be:
```python
data_dict = {
    "fields": fields_tensor,      # shape: [B, N_t, N_x, N_y, N_z, 2]
    "coordinates": coords_tensor, # shape: [N_x, N_y, N_z, 3]
    "time": time_vector           # shape: [N_t]
    "conditions" U_vector         # shape: [B, N_para]
}
```
For 2-D problems set N_z = 1; for 1-D set N_y = N_z = 1.

## The converted data named “data_dict” and other raw data should be saved in a directory “Dataset”. A README file in the directory introduces:
- The origin of the literature (if any)
- The physical background and case setting (varying parameters or initial conditions)
- The shape of tensors in data_dict, and the corresponding spatial and temporal discretization/step length.

# 3. Training & Evaluating Protocal

## 3.1 Data slicing

- Observation window G_obsshape =[B, N_obs, Nx, Ny, Nz, F]
- Forecast target G_tarshape =[B, N_fore, Nx, Ny, Nz, F]
When needed, spatial down-sampling is executed only on G_obs; the target retains native resolution.

## 3.2 Training process

Considering we also have baseline methods (such as POD, FNO, two-stage CNN-based-AE, etc.), we set multiple train and evaluate modules to accommodate the settings for different methods.
Each baseline lives in its own sub-module under src/baselines/.

Before training, update all the tunable hyperparameters in a configuration file named by “{Net_Name}_config” and save it in folder “Save_config_files”.
During training process, train.py for each method will execute the following steps:
- Use dataloading.py to load the dataset, convert them into tensors, and conduct downsampling if necessary.
- Import necessary neural network modules from models.py to define the model.
- Conduct the training and testing loop, during which:
    (1) in training stage, run the model, calculate and store the loss, and update the parameters in the model;
    (2) in testing stage, run the model, calculate the loss;
    (3) according to a pre-defined N_Monitor, print and document all the loss terms in a csv file, stored in “Save_loss_files” folder, named by “loss_log_{Net_Name}.csv”, and  update the loss curve figure named by “loss_history_{Net_Name}.png”;
    (4) compare the improvement of the model, save the best model using state_dict in “Output_Net” folder named by “{Net_Name}.pth”, and perform early-stopping to terminate the    training;
    (5) for reduced-order modeling, if required (SAVE_TRACT == True), process the data in test set and store the latent trajectories;

# 4. Evaluation & Visualisation

After training, we to load the dataset (including additional data not used in training or testing), pick the desired case, load the model and save the results, then plot the field distributions and errors.

 Specifically, in evaluate.py, the general procedures are 
- Load the dataset, hyperparameters and checkpoint of the desired model.
- Extract input data and run the model to create a reconstructed spatiotemporal distribution.
- Calculate MSE loss, plot and save the ground truth, prediction and error fields at desired time steps (default as All steps) as png file in folder “Save_reconstructed_files” named by “Case_{Case_num}_Net_{Net_Name}_MSE_{mse:.3e}”.

In the utils, we will define the general functions to save and plot field distribution figures for 1-D to 3-D datasets.











