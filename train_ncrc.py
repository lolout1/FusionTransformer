import torch
import numpy as np
from Make_Dataset import Poses3d_Dataset, Utd_Dataset
from torch.optim.lr_scheduler import ReduceLROnPlateau
import PreProcessing_ncrc
from Models.model_crossview_fusion import ActTransformerMM
# from Tools.visualize import get_plot
import pickle
from asam import ASAM, SAM
from timm.loss import LabelSmoothingCrossEntropy
import os

exp = 'myexp-utd' #Assign an experiment id

if not os.path.exists('exps/'+exp+'/'):
    os.makedirs('exps/'+exp+'/')
PATH='exps/'+exp+'/'

#CUDA for PyTorch
print("Using CUDA....")

use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")
torch.backends.cudnn.benchmark = True

# Parameters
print("Creating params....")
params = {'batch_size':8,
          'shuffle': True,
          'num_workers': 0}
max_epochs = 200

# Generators
#pose2id,labels,partition = PreProcessing_ncrc_losocv.preprocess_losocv(8)
# pose2id, labels, partition = PreProcessing_ncrc.preprocess()

print("Creating Data Generators...")
dataset = 'utd'
mocap_frames = 100
acc_frames = 150
num_joints = 20 
num_classes = 27

if dataset == 'ncrc':
    training_set = Poses3d_Dataset( data='ncrc',list_IDs=partition['train'], labels=tr_labels, pose2id=tr_pose2id, mocap_frames=mocap_frames, acc_frames=acc_frames, normalize=False)
    training_generator = torch.utils.data.DataLoader(training_set, **params) #Each produced sample is  200 x 59 x 3

    validation_set = Poses3d_Dataset(data='ncrc',list_IDs=partition['test'], labels=valid_labels, pose2id=valid_pose2id, mocap_frames=mocap_frames, acc_frames=acc_frames ,normalize=False)
    validation_generator = torch.utils.data.DataLoader(validation_set, **params) #Each produced sample is 6000 x 229 x 3

else:
    training_set = Utd_Dataset('/home/bgu9/Fall_Detection_KD_Multimodal/data/UTD_MAAD/train_data.npz')
    training_generator = torch.utils.data.DataLoader(training_set, **params)

    validation_set = Utd_Dataset('/home/bgu9/Fall_Detection_KD_Multimodal/data/UTD_MAAD/valid_data.npz')
    validation_generator = torch.utils.data.DataLoader(validation_set, **params)


#Define model
print("Initiating Model...")
model = ActTransformerMM(device = device, mocap_frames=mocap_frames, acc_frames=acc_frames, num_joints=num_joints, in_chans=3, acc_coords=3,
                                  acc_features=1, spatial_embed=32,has_features = False,num_classes=num_classes)
model = model.to(device)


print("-----------TRAINING PARAMS----------")
#Define loss and optimizer
lr=0.001
wt_decay=5e-4

criterion = torch.nn.CrossEntropyLoss()

optimizer = torch.optim.Adam(model.parameters(), lr=lr,weight_decay=wt_decay)
#optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wt_decay)

#ASAM
rho=0.5
eta=0.01
minimizer = ASAM(optimizer, model, rho=rho, eta=eta)

#Learning Rate Scheduler
#scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(minimizer.optimizer, max_epochs)
#print("Using cosine")

#TRAINING AND VALIDATING
epoch_loss_train=[]
epoch_loss_val=[]
epoch_acc_train=[]
epoch_acc_val=[]

#Label smoothing
#smoothing=0.1
#criterion = LabelSmoothingCrossEntropy(smoothing=smoothing)
#print("Loss: LSC ",smoothing)

best_accuracy = 0.
model.load_state_dict(torch.load('/home/bgu9/Fall_Detection_KD_Multimodal/exps/myexp-utd/myexp-utd_best_ckptutdmm.pt'))
# scheduler = ReduceLROnPlateau(optimizer, 'min', verbose = True, patience = 10)

print("Begin Training....")
for epoch in range(max_epochs):
    # Train
    model.train()
    loss = 0.
    accuracy = 0.
    cnt = 0.
    for inputs, targets in training_generator:
        inputs = inputs.to(device); #print("Input batch: ",inputs)
        targets = targets.to(device)

        optimizer.zero_grad()

        # Ascent Step
        #print("labels: ",targets)
        out, logits,predictions = model(inputs.float())
        #print("predictions: ",torch.argmax(predictions, 1) )
        batch_loss = criterion(predictions, targets)
        batch_loss.mean().backward()
        minimizer.ascent_step()

        # Descent Step
        _,_,des_predictions = model(inputs.float())
        criterion(des_predictions, targets).mean().backward()
        minimizer.descent_step()

        with torch.no_grad():
            loss += batch_loss.sum().item()
            accuracy += (torch.argmax(predictions, 1) == targets).sum().item()
        cnt += len(targets)
    loss /= cnt
    accuracy *= 100. / cnt
    print(f"Epoch: {epoch}, Train accuracy: {accuracy:6.2f} %, Train loss: {loss:8.5f}")
    epoch_loss_train.append(loss)
    epoch_acc_train.append(accuracy)
    #scheduler.step()

    #accuracy,loss = validation(model,validation_generator)
    #Test
    model.eval()
    val_loss = 0.
    accuracy = 0.
    cnt = 0.
    model=model.to(device)
    with torch.no_grad():
        for inputs, targets in validation_generator:

            b = inputs.shape[0]
            inputs = inputs.to(device); #print("Validation input: ",inputs)
            targets = targets.to(device)
            
            _,_,predictions = model(inputs.float())
            with torch.no_grad():
                val_loss += batch_loss.sum().item()
                accuracy += (torch.argmax(predictions, 1) == targets).sum().item()
            cnt += len(targets)
        val_loss /= cnt
        accuracy *= 100. / cnt
        
        if best_accuracy < accuracy:
            best_accuracy = accuracy
            torch.save(model.state_dict(),PATH+exp+'_best_ckptafter70.pt'); 
            print("Check point "+PATH+exp+'_best_ckptafter70.pt'+ ' Saved!')

    print(f"Epoch: {epoch},Test accuracy:  {accuracy:6.2f} %, Test loss:  {loss:8.5f}")


    epoch_loss_val.append(loss)
    epoch_acc_val.append(accuracy)


data_dict = {'train_accuracy': epoch_acc_train, 'train_loss':epoch_loss_train, 'val_acc': epoch_loss_val, 'val_loss' : epoch_loss_val }
df.to_csv('loss_acc.csv')

print(f"Best test accuracy: {best_accuracy}")
print("TRAINING COMPLETED :)")

#Save visualization
# get_plot(PATH,epoch_acc_train,epoch_acc_val,'Accuracy-'+exp,'Train Accuracy','Val Accuracy','Epochs','Acc')
# get_plot(PATH,epoch_loss_train,epoch_loss_val,'Loss-'+exp,'Train Loss','Val Loss','Epochs','Loss')
