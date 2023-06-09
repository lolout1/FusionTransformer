import torch
import numpy as np
import pandas as pd
from Make_Dataset import Poses3d_Dataset, Utd_Dataset
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingWarmRestarts
import PreProcessing_ncrc
from Models.model_crossview_fusion import ActTransformerMM
from Models.model_acc_only import ActTransformerAcc
from Models.model_skeleton_only import ActRecogTransformer
from loss import FocalLoss
# from Tools.visualize import get_plot
import pickle
import optuna
from asam import ASAM, SAM
# from timm.loss import LabelSmoothingCrossEntropy
import os
import json

def objective(trial):
    # batch_size = trial.suggest_int('batch_size', 8, 32, step = 8)
    adepth = trial.suggest_int('adepth', 1, 4, step = 1)
    attn_drop_rate = trial.suggest_uniform('attn_drop_rate', .1, .5)
    drop_rate = trial.suggest_uniform('drop_rate', .1, .5)
    num_heads = trial.suggest_int('num_heads', 1, 6, step = 2)
    acc_embed = trial.suggest_categorical('acc_embed', [8,16,32])
    lr = trial.suggest_loguniform('lr', 1e-5, 1e-1)
    wt_decay = trial.suggest_loguniform('wt_decay', 5e-4, 1e-2)
    batch_size = trial.suggest_categorical('batch_size', [8, 16, 32])
    alpha = trial.suggest_uniform('alpha')
    gamma = trail.suggest_categorical('gamma' , [1, 2, 4, 6,8])


    exp = 'ncrcacc-wokd' #Assign an experiment id
    #exp = 'skeletonwithoutkd-utd'

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
    params = {'batch_size':batch_size,
            'shuffle': True,
            'num_workers': 0}
    max_epochs = 100

    print("Creating Data Generators...")
    dataset = 'ncrc'
    mocap_frames = 600
    acc_frames = 150
    num_joints = 29 
    num_classes = 6
    acc_features = 18

    # dataset = 'utd'
    # mocap_frames = 100
    # acc_frames = 150
    # num_joints = 20 
    # num_classes = 27
    # acc_features = 1

    if dataset == 'ncrc':
        tr_pose2id,tr_labels,valid_pose2id,valid_labels,pose2id,labels,partition = PreProcessing_ncrc.preprocess()
        training_set = Poses3d_Dataset( data='ncrc',list_IDs=partition['train'], labels=tr_labels, pose2id=tr_pose2id, mocap_frames=mocap_frames, acc_frames=acc_frames, normalize=True, has_features = False)
        training_generator = torch.utils.data.DataLoader(training_set, **params) #Each produced sample is  200 x 59 x 3

        validation_set = Poses3d_Dataset(data='ncrc',list_IDs=partition['valid'], labels=valid_labels, pose2id=valid_pose2id, mocap_frames=mocap_frames, acc_frames=acc_frames ,normalize=True, has_features = False)
        validation_generator = torch.utils.data.DataLoader(validation_set, **params) #Each produced sample is 6000 x 229 x 3

        test_set = Poses3d_Dataset(data='ncrc',list_IDs=partition['test'], labels=labels,pose2id=pose2id, mocap_frames=mocap_frames, acc_frames=acc_frames ,normalize=False,  has_features = False)
        test_generator = torch.utils.data.DataLoader(test_set, **params) #Each produced sample is 6000 x 229 x 3

    else:
        training_set = Utd_Dataset('/home/bgu9/Fall_Detection_KD_Multimodal/data/UTD_MAAD/train_data.npz')
        training_generator = torch.utils.data.DataLoader(training_set, **params)

        validation_set = Utd_Dataset('/home/bgu9/Fall_Detection_KD_Multimodal/data/UTD_MAAD/valid_data.npz')
        validation_generator = torch.utils.data.DataLoader(validation_set, **params)


    #Define model
    print("Initiating Model...")
    # model = ActTransformerMM(device = device, mocap_frames=mocap_frames, acc_frames=acc_frames, num_joints=num_joints, in_chans=3, acc_coords=3,
    #                                   acc_features=1, spatial_embed=32,has_features = False,num_classes=num_classes)
    #model = ActRecogTransformer( device='cpu', mocap_frames=mocap_frames, num_joints=num_joints,  num_classes=num_classes)
    model = ActTransformerAcc(adepth = adepth,has_features=False, num_heads=num_heads,attn_drop_rate = attn_drop_rate,drop_rate = drop_rate, device = device,
                             acc_frames=acc_frames, num_joints = num_joints, in_chans = 3 , acc_features = acc_features,num_classes=num_classes)
    model = model.to(device)


    print("-----------TRAINING PARAMS----------")
    #Define loss and optimizer
    criterion = FocalLoss(alpha=alpha, gamma=gamma)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr,weight_decay=wt_decay)
    scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=5, T_mult=1, eta_min= 2.5e-05,verbose=True)
    best_accuracy = 0

    print("Begin Training....")
    for epoch in range(max_epochs):
        # Train
        model.train()
        pred = []
        loss = 0.
        accuracy = 0.
        cnt = 0.
        pred_list = []
        target_list = []
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
                pred.extend(torch.argmax(predictions,1).tolist())
                accuracy += (torch.argmax(predictions, 1) == targets).sum().item()
            cnt += len(targets)
        loss /= cnt
        accuracy *= 100. / cnt
        # print('---Train---')
        # for item1 , item2 in zip(target_list, pred_list):
        #     print(f'{item1} | {item2}')
        scheduler.step()
        print(f"Epoch: {epoch}, Train accuracy: {accuracy:6.2f} %, Train loss: {loss:8.5f}")
        # epoch_loss_train.append(loss)
        # epoch_acc_train.append(accuracy)
        #scheduler.step()

        # accuracy,loss = validation(model,validation_generator)
        # Test
        model.eval()
        val_loss = 0.
        val_acc = 0.
        cnt = 0.
        val_pred_list = []
        val_trgt_list = []
        # model=model.to(device)
        with torch.no_grad():
            for inputs, targets in test_generator:

                b = inputs.shape[0]
                inputs = inputs.to(device); #print("Validation input: ",inputs)
                targets = targets.to(device)
                
                _,_,predictions = model(inputs.float())
                with torch.no_grad():
                    val_loss += batch_loss.sum().item()
                    val_acc += (torch.argmax(predictions, 1) == targets).sum().item()
                    
                val_pred_list.extend(torch.argmax(predictions, 1))
                val_trgt_list.extend(targets)

                loss = criterion(predictions,targets)
                val_loss += loss.sum().item()
                val_acc += (torch.argmax(predictions, 1) == targets).sum().item()
                cnt += len(targets)
            val_loss /= cnt
            val_acc *= 100. / cnt
            # print('---Val---')
            # for item1 , item2 in zip(val_trgt_list, val_pred_list):
            #         print(f'{item1} | {item2}')
            # if best_accuracy < accuracy:
            #     best_accuracy = accuracy
                # torch.save(model.state_dict(),PATH+'ncrcacc_woKD.pt')
                # print("Check point "+PATH+'ncrcacc_woKD.pt'+ ' Saved!')

        print(f"Epoch: {epoch},Valid accuracy:  {val_acc:6.2f} %, Valid loss:  {val_loss:8.5f}")

        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()
        return val_acc

    #     epoch_loss_val.append(loss)
    #     epoch_acc_val.append(accuracy)


    # data_dict = {'train_accuracy': epoch_acc_train, 'train_loss':epoch_loss_train, 'val_acc': epoch_acc_val, 'val_loss' : epoch_loss_val }
    # df = pd.DataFrame(data_dict)
    # with open('utdskeleton_woKD_worand.csv'):
    #     df.to_csv('utdskeleton_woKD_worand.csv')

    # print(f"Best test accuracy: {best_accuracy}")
    # print("TRAINING COMPLETED :)")

    #Save visualization
    # get_plot(PATH,epoch_acc_train,epoch_acc_val,'Accuracy-'+exp,'Train Accuracy','Val Accuracy','Epochs','Acc')
    # get_plot(PATH,epoch_loss_train,epoch_loss_val,'Loss-'+exp,'Train Loss','Val Loss','Epochs','Loss')

if __name__ == "__main__":
    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials = 100)
    file_path = 'best_param.txt'
    with open(file_path, 'w') as file:
        json.dump(study.best_params, file)
    for key, value in study.best_params.items():
        print(f'{key}: {value}')