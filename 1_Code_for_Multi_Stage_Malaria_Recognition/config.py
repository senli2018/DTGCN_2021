import os
import torchvision.transforms as transforms

#Path settings for Training and Testing Data
source_path = r'../DTGCN_DATA/1_Multistage_Malaria_Parasite_Recognition/train'
target_path = r'../DTGCN_DATA/1_Multistage_Malaria_Parasite_Recognition/test'



# dir
model_dir = 'Model_and_Log'
net_dir = 'ResNet'
model_name = 'MicroData'
model_path = 'model.pkl'

best_epoch = 'best_epoch'

model_file = 'model.pt'
train_loss_file = 'train_loss.txt'
train_acc_file = 'train_acc.txt'
test_loss_file = 'test_loss.txt'
test_acc_file = 'test_acc.txt'

class_num = 6

# resnet
source_batch_size = 30
target_batch_size = 10


# GCN_model
features_dim_num = 2048
GCN_hidderlayer_dim_num = 512

features_dim = 1024


epoches = 500
k = 10

train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    # transforms.RandomCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

test_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])











