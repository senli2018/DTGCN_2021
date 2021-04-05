# DTGCN

This page shows the original code and data samples of the paper 'Multi-Stage Malaria Parasites Recognition by Deep Transfer Graph Convolutional Network'.


The image samples can be downloaded in [Dataset](https://data.mendeley.com/datasets/2y232dgw36/draft?a=1ab70b62-66c1-4e06-923c-8a86c4deaca0).

The code in this page is for three tasks of multi-stage malaria parasite recognition, large-scale malaria parasite recognition, and babesia parasite recognition. To conduct experiments, please revised the parameters of 'source_path' and 'target_path' in 'config.py' by related pathes of training and testing data for each task .


#Detailed Setting
We have released the code under a OSI compliant license (MIT) with a license file in GitHub (https://github.com/senli2018/DTGCN) and mentioned in our paper.

The code and trained model can be downloaded from GitHub, and the detail information is described below.

Running Environment

	Operating System: Windows 10;

	GPU: Nvidia Geforce 2080Ti GPU;

	Deep learning framework: PyTorch 1.0.0 in Python 3.6.0;

Requirements:

	Pytorch 1.0.0;

	Torchvision 0.4.1;

	Scipy 1.1.0;

	Numpy 1.17.4;

Parameter in code has been described in the GitHub, and can be directly run. The detail parameters are introduced as follow:

In this section, the batch size is set as 15 for source and target data with learning rate of 1e-5 which multiply 0.1 in every 50 iterations.

For the training details in GCN, the two graph convolution layers are trained with the output CNN features of the pre-trained CNN. 

For both of CNN and GCN training, the DTGCN is implemented by the PyTorch framework with GTX2080Ti GPU and employ Adam optimizer to optimize the parameters of the network. As for GCN, the learning rate is 2e-7 and is optimized by manually gradient descent operation. And the settings of large-scale malaria parasites recognition are also following this setting.

Finally, we introduce how to run DTGCN code.
1.	Download the DTGCN code from GitHub (https://github.com/senli2018/DTGCN).

2.	Download the dataset from Mendeley and release it into the root dir of code.

3.	The data consist three tasks of ‘1_multistage_malaria_classification’, ‘2_Unseen_Malaria_classification’ and ‘3_babesia_classification’, it can change the path variables in ‘config.py’, such as 

(a)	Multi-stage malaria parasites recognition task:

source_dataset_path = '1_multistage_malaria_classification/train'

target_dataset_path = '1_multistage_malaria_classification/test'	

(b)	Large scale malaria classification task:

source_dataset_path = '2_Unseen_Malaria_classification/train'

target_dataset_path = '2_Unseen_Malaria_classification/test'	

(c)	babesia_classification

source_dataset_path = '3_babesia_classification '

target_dataset_path = '3_babesia_classification '

4.Taking multi-stage malaria parasites recognition task as an example, just run ‘main.py’, and the programme will automatically load existing models and start train the GCN, with outputting loss and predicted results along with epochs. It will be soon convergence in 10 epochs, and reported results can be obtained.
