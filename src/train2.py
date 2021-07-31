#from data_utils import EarlyStopping
from unet import UNet
from preprocess import tensorize_image, tensorize_mask, image_mask_check#Functions in preprocess file were imported
import os
import glob
import numpy as np
import torch.nn as nn
import tqdm
import torch
from draw_graph import draw_graph
from adabound import AdaBound 
from test import predict



######### PARAMETERS ##########
valid_size = 0.2#Validation dataset is used to evaluate a particular model, but this is for frequent evaluation.
test_size  = 0.1#rate of data to be tested
batch_size = 4#it means how many data the model will process at the same time.
epochs = 35#Epoch count is the number of times all training data is shown to the network during training.
cuda =True
input_shape = (224, 224)#What size will the image resize
n_classes = 2
###############################


AUG_IMAGE='../data/augmentation'
AUG_MASK='../data/augmentation_mask'
IMAGE_DIR = '../data/images'
MASK_DIR='../data/masks'



# PREPARE AND SHUFFLE IMAGE AND MASK LISTS
image_path_listt = glob.glob(os.path.join(IMAGE_DIR, '*'))
image_path_listt.sort()

mask_path_listt = glob.glob(os.path.join(MASK_DIR, '*'))
mask_path_listt.sort()

data=list(zip(image_path_listt,mask_path_listt))

np.random.shuffle(data)

image_path_listt, mask_path_listt = zip(*data)

image_path_list=list(image_path_listt)
mask_path_list=list(mask_path_listt)

# PREPARE AND SHUFFLE AUG IMAGE AND MASK LISTS
aug_path_list = glob.glob(os.path.join(AUG_IMAGE, '*'))
aug_path_list.sort()
aug_mask_path_list = glob.glob(os.path.join(AUG_MASK, '*'))
aug_mask_path_list.sort()





# DATA CHECK
image_mask_check(image_path_list, mask_path_list)
#Checked whether the elements in mask_path_list and image_path_list list are the same.



# SHUFFLE INDICES
indices = np.random.permutation(len(image_path_list))
#A random array of permutations for the length of the image_path_list steps_per_epoch = len (train_input_path_list) // batch_size is created


# DEFINE TEST AND VALID INDICES
test_ind  = int(len(indices) * test_size)#Multiply indices length by test_size and assign it to an int-shaped variable
valid_ind = int(test_ind + len(indices) * valid_size)

# SLICE TEST DATASET FROM THE WHOLE DATASET
test_input_path_list = image_path_list[:test_ind]
test_label_path_list = mask_path_list[:test_ind]

# SLICE VALID DATASET FROM THE WHOLE DATASET
valid_input_path_list = image_path_list[test_ind:valid_ind]#Get 476 to 1905 elements of the image_path_list list
valid_label_path_list = mask_path_list[test_ind:valid_ind]#Get 476 to 1905 elements of the mask_path_list list

# SLICE TRAIN DATASET FROM THE WHOLE DATASET
train_input_path_list = image_path_list[valid_ind:]#Get the elements of the image_path_list list from 1905 to the last element
train_label_path_list = mask_path_list[valid_ind:]#Get the elements of the mask_path_list list from 1905 to the last element




aug_size=int(len(aug_mask_path_list)/2)
input_train=aug_path_list[:aug_size]+train_input_path_list+aug_path_list[aug_size:]
label_train=aug_mask_path_list[:aug_size]+train_label_path_list+aug_mask_path_list[aug_size:]



#Shuffle with aug dataset
all_data=list(zip(input_train,label_train))
np.random.shuffle(all_data)
input_train, label_train = zip(*all_data)
train_input_path_list=list(input_train)
train_label_path_list=list(label_train)






steps_per_epoch = len(train_input_path_list)//batch_size
# Find how many times to do it by dividing the length of the train data (training data) by batch_size
#in an epoch, a data string in the dataset goes to the end in neural networks
#It then waits there until the batch reaches you, the error rate is calculated after the data reaches the end
#Divide the training data set by 4 since our batch_size is 4

# CALL MODEL
model = UNet(n_channels=3, n_classes=2, bilinear=True)
#Enter parameters into model and assign output to variable

# DEFINE LOSS FUNCTION AND OPTIMIZER
criterion = nn.BCELoss()#Creates a criterion that measures the Binary Cross Entropy between target and output:
#BCELoss is an acronym for Binary CrossEntropyLoss, a special case of BCOMoss CrossEntropyLoss used only for two categories of problems.
optimizer = AdaBound(model.parameters(), lr=1e-4, final_lr=0.1)
#early_stopping = EarlyStopping()

    
# IF CUDA IS USED, IMPORT THE MODEL INTO CUDA
if cuda:
    model = model.cuda()

val_losses=[]
train_losses=[]
# TRAINING THE NEURAL NETWORK
for epoch in tqdm.tqdm(range(epochs)):

    running_loss = 0
    #In each epoch, images and masks are mixed randomly in order not to output images sequentially.
    pair_IM=list(zip(train_input_path_list,train_label_path_list))
    np.random.shuffle(pair_IM)
    unzipped_object=zip(*pair_IM)
    zipped_list=list(unzipped_object)
    train_input_path_list=list(zipped_list[0])
    train_label_path_list=list(zipped_list[1])
    
    for ind in range(steps_per_epoch):
        batch_input_path_list = train_input_path_list[batch_size*ind:batch_size*(ind+1)]
        batch_label_path_list = train_label_path_list[batch_size*ind:batch_size*(ind+1)]
        batch_input = tensorize_image(batch_input_path_list, input_shape, cuda)
        batch_label = tensorize_mask(batch_label_path_list, input_shape, n_classes, cuda)
        
        
        optimizer.zero_grad()

        outputs = model(batch_input) 
        

        
        loss = criterion(outputs, batch_label)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        
        #validation 
        if ind == steps_per_epoch-1:
            
            train_losses.append(running_loss)
            print('training loss on epoch {}: {}'.format(epoch, running_loss))

            val_loss = 0
            for (valid_input_path, valid_label_path) in zip(valid_input_path_list, valid_label_path_list):
                batch_input = tensorize_image([valid_input_path], input_shape, cuda)
                batch_label = tensorize_mask([valid_label_path], input_shape, n_classes, cuda)
                outputs = model(batch_input)
                loss = criterion(outputs, batch_label)
                val_loss += loss.item()
                val_losses.append(val_loss)
                break  
            print('validation loss on epoch {}: {}'.format(epoch, val_loss))
# =============================================================================
#     early_stopping(val_loss)
#     if early_stopping.early_stop:
#         break
# =============================================================================
            
        
        

torch.save(model.state_dict(), '../data/models/8000_model2.pth')            
print("Model Saved!")



draw_graph(val_losses,train_losses,epochs)
predict(test_input_path_list)
