# MDGAE-DTI
MDGAE-DTI: drug-target interaction prediction based on multi-information integration and graph autoencoder.

The model utilized multiple similarity matrices of targets and drugs to portray the drug similarity and target similarity at different approaches, after which these similarity matrices are integrated, and then the feature vectors of drug nodes and target nodes are generated in a graph neural network using node similarity as initial features, and a multi-layer perceptron is applied to update the features of the nodes. The final representations of drugs and target proteins are incorporated into a bilinear decoder to complete the task of DTI prediction.

# Requirment
The main requirements are:
- numpy==1.19.2
- mxnet-cu102==1.6.0.post0
- dgl-cu102==0.6.0.post1
 use`conda env create -f environment.yaml` to set up the environment.

Overview - `data/` contains the necessary dataset files
