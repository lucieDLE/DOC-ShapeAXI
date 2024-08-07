
# Automated 3D shape classification 

"3DShapeClassification" is an extension for 3D Slicer, a free, open-source software for medical, biomedical and related imaging research. This extension aims to provide a Graphical User Interface for a deep-learning classification tool that we developed at the University of North Carolina in Chapel Hill in collaboration with the University of Michigan in Ann Arbor. This tool takes ______ and automatically classify them using [ShapeAXI](https://github.com/DCBIA-OrthoLab/ShapeAXI/) framework.

# Requirements

 - This extension works with Linux and Windows.

# How to use the extension
 
## Installation

<!-- You can download the extension on the Slicer Extension Manager. Slicer needs to restart after installation. -->


## Running the module

 - You will find the module under the name  `ShapeClassification` in the `Classification` tab.
 #### Inputs group
 - Data Type: type of dataset to predict --> three data types are available for now: Cleft/Airways/Condyles
 - The input file must be a 
 - surf column: column name in the input file containing path to data (default: surf)
   This should usually be set to 320px.
 - mount point: 
 #### Output group
 - output directory: 
 #### Model group:
 There are two mode available: using your trained model with specific model architecture, or use the models available on github. 

 For the first mode, you will need to provide the network type and the model.
 - Network type: the type of architecture used to train the model. There are three architectures available: Saxi, SaxiRing and SaxiMHAFB. visit [ShapeAXI](https://github.com/DCBIA-OrthoLab/ShapeAXI/) for more information.
 - Model for prediction: path to the neural network model trained with ShapeAxi.

For the second mode, you only need to click on the check box and it will automatically select the best model available.
---> Description of the models for each dataset
