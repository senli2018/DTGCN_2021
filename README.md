# DTGCN

This page shows the original code and data samples of the paper 'Multi-Stage Malaria Parasites Recognition by Deep Transfer Graph Convolutional Network'.


The image samples can be downloaded in [Dataset](https://data.mendeley.com/datasets/2y232dgw36/draft?a=1ab70b62-66c1-4e06-923c-8a86c4deaca0).

The code in this page is for three tasks of multi-stage malaria parasite recognition, large-scale malaria parasite recognition, and babesia parasite recognition. To conduct experiments, please revised the parameters of 'source_path' and 'target_path' in 'config.py' by related pathes of training and testing data for each task .

Taking multi-stage malaria parasite recognition task as an example, its experiment is implemented as follows,

1. Download the data from  [Mendeley](https://data.mendeley.com/datasets/2y232dgw36/draft?a=1ab70b62-66c1-4e06-923c-8a86c4deaca0), and release it. 
2. Revise the parameters source_path=r'real training data path of multi-stage malaria recognition' and target_path =r'real testing data path of multi-stage malaria recognition' in 'config.py'. 
3. Run train.py to begin the code execution, which will automatically train the model and output the testing results along with training process.

#Detailed Setting
We have released the code under a OSI compliant license (MIT) with a license file in GitHub (https://github.com/senli2018/DTGCN_2021) and mentioned in our paper.

Besides, the running environment and requirements are described below.

Running Environment

	Operating System: Windows 10;

	GPU: Nvidia Geforce 2080Ti GPU;

	Deep learning framework: PyTorch 1.0.0 in Python 3.6.0;

Requirements:

	Pytorch 1.0.0;

	Torchvision 0.4.1;

	Scipy 1.1.0;

	Numpy 1.17.4;

Parameters in our method has been set in the code, and can be directly run. 

Besides, we also release the codes of all compared methods in our paper, including VggNet, ResNet, GoogLeNet, Reference 22-25, Baseline, Ours+KNN, Ours+ResNet34, Ours+Res50, and evaluation of train_size. These codes are compressed in the 'compared methods.zip', and their executions are also following the steps of DTGCN code, with revising the data pathes in 'config.py'.

Note that, this paper conducts three types of experiments, including multi-stage malaria parasite recognition, large scale malaria parasites recognition, and Babesia parasite recognition. The experiments of DTGCN and compared methods utilize the same training and testing images in each type of experiments, and the details are summarized in the attached Table 1.

![image](https://github.com/senli2018/DTGCN_2021/blob/main/Table_1.jpg)
