# SGCN-CPI
########################################################## 

The source code for

SGCN-CPI: Integrating Retrieved Similar Compound-Protein Pairs in Interaction Affinity Prediction using Semi-supervised Graph Convolutional Network
##########################################################

****Requirements**
**

Python 3.8

Tensorflow 0.12.0

Keras (1.1 or higher)

Scipy 1.3.2

numpy

keras-gcn (download and install it-downloadlink:https://github.com/tkipf/keras-gcn) 
##########################################################

****Data**
**

Download the data from the following link

https://drive.google.com/open?id=1B72WnWMbywxK2M9RntquRWQ3cm6U9YoW

Download the folded data from the following link:

https://drive.google.com/open?id=15KotSJWknMOAnHM68RpOh_rqMISsMwsE
##########################################################

****Usage**
**
At first, keras-gcn should be installed. The hyperparameters are set in config.py file. In this file, you could select Davis, KIBA, and BindingDB datasets. Also, The other hyperparameters could be set to your desired values. Then you can run the SGCN_CPI.py.
