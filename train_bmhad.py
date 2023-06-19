import torch
import numpy as np
import pandas as pd
from Make_Dataset import Berkley_mhad
from torch.optim.lr_scheduler import CosineAnnealingLR
from Models.model_acc_bmhad import ActTransformerAcc
from loss import FocalLoss
# from Tools.visualize import get_plot
import pickle
# from timm.loss import LabelSmoothingCrossEntropy
import os

exp = 'bmad' #Assign an experiment id

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
<<<<<<< HEAD
params = {'batch_size':32,
=======
params = {'batch_size':64,
>>>>>>> 4318cad5f5e013e19f76d4b929e3232acb2117f8
          'shuffle': True,
          'num_workers': 0}
max_epochs = 200

# Generators
#pose2id,labels,partition = PreProcessing_ncrc_losocv.preprocess_losocv(8)
# pose2id, labels, partition = PreProcessing_ncrc.preprocess()

print("Creating Data Generators...")
acc_frames = 512
num_classes = 36
num_heads = 8
adepth = 4

exp_id = f'{exp}d{adepth}h{num_heads}'

training_set = Berkley_mhad('/Users/tousif/Lstm_transformer/Fall_Detection_KD_Multimodal/data/berkley_mhad/berkley_mhad_train.npz')
training_generator = torch.utils.data.DataLoader(training_set, **params)

valid_set = Berkley_mhad('/Users/tousif/Lstm_transformer/Fall_Detection_KD_Multimodal/data/berkley_mhad/berkley_mhad_val.npz')
validation_generator= torch.utils.data.DataLoader(valid_set, **params)


#Define model
print("Initiating Model...")
# model = ActTransformerMM(device = device, mocap_frames=mocap_frames, acc_frames=acc_frames, num_joints=num_joints, in_chans=3, acc_coords=3,
#                                   acc_features=1, spatial_embed=16,has_features = False,num_classes=num_classes, num_heads=8)

model = ActTransformerAcc(adepth = adepth,device= device, acc_frames= acc_frames,has_features=False, num_heads = num_heads, num_classes=num_classes)
model = model.to(device)


print("-----------TRAINING PARAMS----------")
#Define loss and optimizer
lr=0.01
wt_decay=1e-3
# class_weights = torch.reciprocal(torch.tensor([74.23, 83.87, 56.75, 49.78, 49.05, 93.92]))
criterion = torch.nn.CrossEntropyLoss()
# criterion = FocalLoss(alpha=0.25, gamma=2)

optimizer = torch.optim.AdamW(model.parameters(), lr=lr,weight_decay=wt_decay)
#optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wt_decay)
scheduler = CosineAnnealingLR(optimizer=optimizer,T_max=100, eta_min=1e-4,last_epoch=-1,verbose=True)

#ASAM
# rho=0.5
# eta=0.01
# minimizer = ASAM(optimizer, model, rho=rho, eta=eta)

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

best_accuracy = 0
# model.load_state_dict(torch.load('/home/bgu9/Fall_Detection_KD_Multimodal/exps/myexp-utd/myexp-utd_best_ckptutdmm.pt'))
# scheduler = ReduceLROnPlateau(optimizer, 'min', verbose = True, patience = 10)

print("Begin Training....")
for epoch in range(max_epochs):
    # Train
    model.train()
    loss = 0.
    accuracy = 0.
    cnt = 0.
    pred_list = []
    target_list = []
    i = 0

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
        optimizer.step()
        # minimizer.ascent_step()

        # # Descent Step
        # _,_,des_predictions = model(inputs.float())
        # criterion(des_predictions, targets).mean().backward()
        # minimizer.descent_step()
        pred_list.extend(torch.argmax(predictions, 1))
        target_list.extend(targets)
        with torch.no_grad():
            loss += batch_loss.sum().item()
            accuracy += (torch.argmax(predictions, 1) == targets).sum().item()
        cnt += len(targets)
<<<<<<< HEAD
        print(cnt)
    loss /= cnt
    accuracy *= 100. / cnt
    
=======
    loss /= cnt
    accuracy *= 100. / cnt
    print(loss)
>>>>>>> 4318cad5f5e013e19f76d4b929e3232acb2117f8
    # print('---Train---')
    # for item1 , item2 in zip(target_list, pred_list):
    #     print(f'{item1} | {item2}')
    scheduler.step()
    print(f"Epoch: {epoch}, Train accuracy: {accuracy:6.2f} %, Train loss: {loss:8.5f}")
    epoch_loss_train.append(loss)
    epoch_acc_train.append(accuracy)
    #scheduler.step()

    # accuracy,loss = validation(model,validation_generator)
    # Test
    model.eval()
    val_loss = 0.
    accuracy = 0.
    cnt = 0.
    val_pred_list = []
    val_trgt_list = []
    # model=model.to(device)
    with torch.no_grad():
        for inputs, targets in validation_generator:

            b = inputs.shape[0]
            inputs = inputs.to(device); #print("Validation input: ",inputs)
            targets = targets.to(device)
            
            _,_,predictions = model(inputs.float())
            val_pred_list.extend(torch.argmax(predictions, 1))
            val_trgt_list.extend(targets)

            loss = criterion(predictions,targets)
            val_loss += loss.sum().item()
            accuracy += (torch.argmax(predictions, 1) == targets).sum().item()
            cnt += len(targets)
        val_loss /= cnt
        accuracy *= 100. / cnt
        # print('---Val---')
        # for item1 , item2 in zip(val_trgt_list, val_pred_list):
        #         print(f'{item1} | {item2}')
        if best_accuracy < accuracy:
            best_accuracy = accuracy
            torch.save(model.state_dict(),PATH+f'{exp_id}.pt')
            print("Check point " + PATH + exp_id + ".pt Saved!")

    print(f"Epoch: {epoch},Valid accuracy:  {accuracy:6.2f} %, Valid loss:  {val_loss:8.5f}")
    epoch_loss_val.append(val_loss)
    epoch_acc_val.append(accuracy)


data_dict = {'train_accuracy': epoch_acc_train, 'train_loss':epoch_loss_train, 'val_acc': epoch_loss_val, 'val_loss' : epoch_acc_val }
df = pd.DataFrame(data_dict)

try:
    file = open(f'{exp_id}.csv', 'x')
    df.to_csv(file, index = False)

except FileExistsError:
    print('Experiment already done')


print(f"Best test accuracy: {best_accuracy}")
print("TRAINING COMPLETED :)")

#Save visualization
# get_plot(PATH,epoch_acc_train,epoch_acc_val,'Accuracy-'+exp,'Train Accuracy','Val Accuracy','Epochs','Acc')
# get_plot(PATH,epoch_loss_train,epoch_loss_val,'Loss-'+exp,'Train Loss','Val Loss','Epochs','Loss')
