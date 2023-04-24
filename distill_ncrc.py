import torch
import numpy as np
from Make_Dataset import Poses3d_Dataset
import torch.nn as nn
import PreProcessing_ncrc
from Models.model_crossview_fusion import ActTransformerMM
from Models.model_acc_only import ActTransformerAcc
# from Tools.visualize import get_plot
from tqdm import tqdm
import torch.nn.functional as F
import pickle
from asam import ASAM, SAM
from timm.loss import LabelSmoothingCrossEntropy
from loss import SemanticLoss
import os



exp = 'myexp-1' #Assign an experiment id

if not os.path.exists('exps/'+exp+'/'):
    os.makedirs('exps/'+exp+'/')
PATH='exps/'+exp+'/'

#CUDA for PyTorch
print("Using CUDA....")

use_cuda = torch.cuda.is_available()
print(use_cuda)
device = torch.device("cuda:0" if use_cuda else "cpu")
torch.backends.cudnn.benchmark = True



# Parameters
print("Creating params....")
params = {'batch_size':8,
          'shuffle': True,
          'num_workers': 0}

num_epochs = 250

# Generators
#pose2id,labels,partition = PreProcessing_ncrc_losocv.preprocess_losocv(8)
pose2id, labels, partition = PreProcessing_ncrc.preprocess()

print("Creating Data Generators...")
mocap_frames = 600
acc_frames = 150
training_set = Poses3d_Dataset( data='ncrc',list_IDs=partition['train'], labels=labels, pose2id=pose2id, mocap_frames=mocap_frames, acc_frames=acc_frames, normalize=False)
training_generator = torch.utils.data.DataLoader(training_set, **params) #Each produced sample is  200 x 59 x 3

validation_set = Poses3d_Dataset(data='ncrc',list_IDs=partition['test'], labels=labels, pose2id=pose2id, mocap_frames=mocap_frames, acc_frames=acc_frames ,normalize=False)
validation_generator = torch.utils.data.DataLoader(validation_set, **params) #Each produced sample is 6000 x 229 x 3

#Define model
print("Initiating Model...")
teacher_model = ActTransformerMM(device)
student_model = ActTransformerAcc(device)

teacher_model.cuda()
student_model.cuda()




#Define loss and optimizer
lr=0.0025
wt_decay=5e-4



#Learning Rate Scheduler
#scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(minimizer.optimizer,(num_epochs)
#print("Using cosine")

#TRAINING AND VALIDATING


#Label smoothing
#smoothing=0.1
#criterion = LabelSmoothingCrossEntropy(smoothing=smoothing)
#print("Loss: LSC ",smoothing)



def train(epoch, num_epochs, student_model, teacher_model, criterion, best_accuracy):
    teacher_model.eval()
    with tqdm(total  = len(training_generator), desc = f'Epoch {epoch}/{num_epochs}',ncols = 128) as pbar:
        # Train
        student_model.train()
        train_loss = 0.
        accuracy = 0.
        cnt = 0.
        alpha = 0.7
        T = 2.0
        for inputs, acc_input, targets in training_generator:
            # Transfering the input, targets to the GPU
            inputs = inputs.to(device); #print("Input batch: ",inputs)
            targets = targets.to(device)
            acc_input = acc_input.to(device)


            optimizer.zero_grad()

            # Ascent Step
            #print("labels: ",targets)

            #Prediction step
            out, student_logits,student_pred = student_model(inputs.float())
            teacher_out, teacher_logits, teacher_pred = teacher_model(inputs.float())
            # detached_pred = predictions.detach()
            # teacher_output = teacher_output.detach()
            #print("predictions: ",torch.argmax(predictions, 1) )
            loss = criterion(student_logits, targets, teacher_logits, T, alpha)
            loss.mean().backward()
            minimizer.ascent_step()

            # Descent Step
            # Not really sure why I need to make prediction again 
            _, des_student_logits,des_stud_pred = student_model(inputs.float())
            _, des_teacher_logits, des_teacher_pred = teacher_model(inputs.float())
            criterion(des_student_logits, targets, des_teacher_logits, T=2.0, alpha = 0.7).mean().backward()
            minimizer.descent_step()

            with torch.no_grad():
                train_loss += loss.sum().item()
                # print(loss)
                # print(type(loss))
                accuracy += (torch.argmax(des_stud_pred, 1) == targets).sum().item()
            cnt += len(targets)

            temp_loss = train_loss / cnt
            temp_acc = (accuracy * 100) / cnt

            pbar.update(1)
            pbar.set_postfix({'train_loss' : temp_loss, 'train_acc' : temp_acc})
            
        train_loss /= cnt
        accuracy *= 100. / cnt
        epoch_loss_train.append(train_loss)
        epoch_acc_train.append(accuracy)

        #Val
        student_model.eval()
        val_loss = 0.
        val_accuracy = 0.
        cnt = 0.
        student_model=student_model.to(device)
        with torch.no_grad():
            for inputs,_, targets in validation_generator:

                b = inputs.shape[0]
                inputs = inputs.to(device); #print("Validation input: ",inputs)
                targets = targets.to(device)
                
                out, student_logits,predictions = student_model(inputs.float())
                loss_score = F.cross_entropy(predictions, targets)
                
                with torch.no_grad():
                    val_loss += loss_score.sum().item()
                    val_accuracy += (torch.argmax(predictions, 1) == targets).sum().item()
                cnt += len(targets)
            val_loss /= cnt
            val_accuracy *= 100. / cnt
            
        print(f"\n Epoch: {epoch},Val accuracy:  {val_accuracy:6.2f} %, Val loss:  {val_loss:8.5f}%")
        if best_accuracy < val_accuracy:
                best_accuracy = val_accuracy
                #need to add arguements here also for different experiments
                torch.save(student_model.state_dict(),PATH+exp+'_best_ckpt.pt'); 
                print("Check point "+PATH+exp+'_best_ckpt.pt'+ ' Saved!')


        epoch_loss_val.append(val_loss)
        epoch_acc_val.append(val_accuracy)



if __name__ == "__main__":
    max_epoch = 250
    best_accuracy = 0 
    epoch_loss_train=[]
    epoch_loss_val=[]
    epoch_acc_train=[]
    epoch_acc_val=[]
    teacher_model.load_state_dict(torch.load('weights/model_crossview_fusion.pt'))
    #Optimizer
    optimizer = torch.optim.SGD(student_model.parameters(), lr=lr, momentum=0.9,weight_decay=wt_decay)
    #optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wt_decay)

    #ASAM
    rho=0.5
    eta=0.01
    minimizer = ASAM(optimizer, student_model, rho=rho, eta=eta)
    
    best_accuracy = 0
    criterion = SemanticLoss()
    #criterion selection using arguements
    total_params = 0
    print("-----------TRAINING PARAMS----------")
    for name , params in student_model.named_parameters():
        total_params += params.numel()
        print(f'Layer {name} | Size: {params.size()} | Params: {params.numel()}')
    print(f'Total parameter: {total_params}')

    for epoch in range(1,max_epoch+1): 
        train(epoch, max_epoch, student_model, teacher_model,criterion,  best_accuracy = best_accuracy )
        best_accuracy = epoch_acc_val[epoch - 1]
        print(best_accuracy)