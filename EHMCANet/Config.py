import os
import torch
import time
import ml_collections

## PARAMETERS OF THE MODEL
save_model = True
tensorboard = True
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
use_cuda = torch.cuda.is_available()
seed = 666
os.environ['PYTHONHASHSEED'] = str(seed)


cosineLR = True # whether use cosineLR or not
n_channels = 3
num_classes=1
n_labels = 1
epochs = 2000
img_size = 256
# img_size = 224
print_frequency = 1
save_frequency = 5000
vis_frequency = 10
early_stopping_patience = 50
learning_rate = 1e-3
pretrain = False

####datasets
task_name = 'MoNuSeg'
# task_name ='Cell_Nucleus'   ###

batch_size = 4

# model_name = 'UNet'
model_name = 'EHMCANet' #

train_dataset = './dataset/'+ task_name+ '/Train_Folder/'
val_dataset = './dataset/'+ task_name+ '/Val_Folder/'
test_dataset = './dataset/'+ task_name+ '/Test_Folder/'

session_name       = 'Test_session' + '_' + time.strftime('%m.%d_%Hh%M')
save_path          = task_name +'/'+ model_name +'/' + session_name + '/'
model_path         = save_path + 'models/'
tensorboard_folder = save_path + 'tensorboard_logs/'
logger_path        = save_path + session_name + ".log"
visualize_path     = save_path + 'visualize_val/'


def get_config():
    config = ml_collections.ConfigDict()
    config.base_channel = 64
    return config

test_session = "Test_session_10.26_23h28"##





