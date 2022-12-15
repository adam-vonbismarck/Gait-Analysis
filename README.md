# CS 1430 Final Project: Gait Analysis
```
  ____       _ _        _                _           _     
 / ___| __ _(_) |_     / \   _ __   __ _| |_   _ ___(_)___ 
| |  _ / _` | | __|   / _ \ | '_ \ / _` | | | | / __| / __|
| |_| | (_| | | |_   / ___ \| | | | (_| | | |_| \__ \ \__ \
 \____|\__,_|_|\__| /_/   \_\_| |_|\__,_|_|\__, |___/_|___/
                                           |___/           
```
     
**Computer vision final project for team "Wallace and Gromit"**
POSTER LINK: https://docs.google.com/presentation/d/1ziSVpj7NRCUMtv_hRPUFRiXd8TEyiPkR8Snf9uUYz_U/edit?usp=sharing

## Prerequisites
- Download CASIA dataset (this can be trimmed to only include relevent images using filescript.py)
- Include path for downloaded dataset in config.py
## How to run code
Run the following command to run the model

    python main.py

By default, this will use the CASIA dataset with minimal information printed in the terminal.

Several parameters can be added to the run_model function. You can set **useCasiaB** to false to use our custom dataset. You can set **useSpecial** to true to test with videos of coats (note that this will not do anything when using the CASIA dataset). You can also set **verbose** to True which wil print verbose outputs from the random forest classifier as well as print a detailed comparison of predicted labels and actual labels.
## Known bugs
On some architectures, the video reading CV2 functionality will not work as expected and cause an error.