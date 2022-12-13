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

**Code structure**

When the classifier is tested, the code is run by first using the methods in preprocess.py. These methods 
extract the human and center them in the middle of the frame. This is done based on the assumption that the
model is only fed black and white images. The image is then cropped to a size defined in the config file.
The images are fed through this after they have been extracted and turned into a numpy array of images for each
person. The images are then fed through either the gait energy image or the gait entropy image methods. These
methods are defined in the GEI.py file. The images are then fed through the classifier which is defined in the
classifier.py file.

**Usage**