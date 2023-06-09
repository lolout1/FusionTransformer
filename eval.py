import torch
import numpy as np
import torch.nn.functional as F
from Make_Dataset import Poses3d_Dataset, Utd_Dataset
import torch.nn as nn
import PreProcessing_ncrc
from Models.model_crossview_fusion import ActTransformerMM
from Models.model_acc_only import ActTransformerAcc
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, f1_score, precision_recall_fscore_support



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


# dataset = 'utd'
# mocap_frames = 100
# acc_frames = 150
# num_joints = 20
# num_classes = 27

dataset = 'ncrc'
mocap_frames = 600
acc_frames = 150
num_joints = 29 
num_classes = 6

if dataset == 'ncrc':
    tr_pose2id,tr_labels,valid_pose2id,valid_labels,pose2id,labels,partition = PreProcessing_ncrc.preprocess()
    training_set = Poses3d_Dataset( data='ncrc',list_IDs=partition['train'], labels=tr_labels, pose2id=tr_pose2id, mocap_frames=mocap_frames, acc_frames=acc_frames,has_features=False, normalize=False)
    training_generator = torch.utils.data.DataLoader(training_set, **params) #Each produced sample is  200 x 59 x 3

    validation_set = Poses3d_Dataset(data='ncrc',list_IDs=partition['valid'], labels=valid_labels, pose2id=valid_pose2id, mocap_frames=mocap_frames, acc_frames=acc_frames,has_features=False,normalize=False)
    validation_generator = torch.utils.data.DataLoader(validation_set, **params) #Each produced sample is 6000 x 229 x 3

    test_set = Poses3d_Dataset(data='ncrc',list_IDs=partition['test'], labels=labels, pose2id=pose2id, mocap_frames=mocap_frames, acc_frames=acc_frames,has_features=False,normalize=False)
    test_generator = torch.utils.data.DataLoader(test_set, **params) #Each produced sample is 6000 x 229 x 3



else:

    test_set = Utd_Dataset('/Users/tousif/Lstm_transformer/randtest_data.npz')
    test_generator = torch.utils.data.DataLoader(test_set, **params)


#
#Define model
print("Initiating Model...")

# student_model = ActTransformerAcc(device = device, acc_frames=150, num_joints=num_joints, in_chans=3, acc_coords=3,
#                                   acc_features=18, has_features =True,num_classes=num_classes)
# student_model.load_state_dict(torch.load('/home/bgu9/Fall_Detection_KD_Multimodal/exps/ncrc/ncrc_ncrc_ckpt_wdistaccurate.pt'))

#teacher_model = ActTransformerMM(device = device, mocap_frames=mocap_frames, acc_frames=150, num_joints=num_joints, in_chans=3, acc_coords=3,
                                  #acc_features=1, spatial_embed=32,has_features = False,num_classes=num_classes)
#teacher_model.load_state_dict(torch.load('/home/bgu9/Fall_Detection_KD_Multimodal/exps/myexp-utd/myexp-utd_best_ckptafter70.pt'))
# student_model.cuda()
teacher_model = ActTransformerAcc(adepth = 4,device= device, acc_frames= acc_frames, num_joints = 29,has_features=False)
teacher_model.load_state_dict(torch.load('exps/ncrc/ncrcacc_woKD.pt'))
teacher_model.to(device=device)

# student_model.eval()
teacher_model.eval()

y_true = []
y_pred = []
val_loss = 0
val_accuracy = 0
val_t_accuracy = 0
cnt = 0.
# student_model=student_model.to(device)
teacher_model = teacher_model.to(device)
with torch.no_grad():
    for inputs,targets in test_generator:
        y_true.extend(targets.numpy().tolist())
        inputs = inputs.to(device); #print("Validation input: ",inputs)
        targets = targets.to(device)
        
        
        out, student_logits,predictions = teacher_model(inputs.float())
        loss_score = F.cross_entropy(predictions, targets)
        y_pred.extend(torch.argmax(predictions, 1).cpu().numpy().tolist())
        with torch.no_grad():
            val_loss += loss_score.sum().item()
            val_accuracy += (torch.argmax(predictions, 1) == targets).sum().item()

        cnt += len(targets)
    val_loss /= cnt
    val_accuracy *= 100. / cnt


# compute the confusion matrix
y_pred = np.array(y_pred)
y_true = np.array(y_true)
precision, recall, f1, _ = precision_recall_fscore_support(y_true=y_true, y_pred=y_pred)
for i in range(len(precision)):
    print(f"Class {i}: Precision={precision[i]}, Recall={recall[i]}, F1-score={f1[i]}")
cm = confusion_matrix(y_true, y_pred)

print("----------OVERALL METRICS--------")
print("Overall Precision: ",np.mean(precision)*100)
print("Overall Recall: ",np.mean(recall)*100)
print("Overall F1-score: ",np.mean(f1)*100)
print(f"Val accuracy:  {val_accuracy:6.2f} %, Val loss:  {val_loss:8.5f}%")

# plot the confusion matrix
plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
plt.colorbar()
plt.xticks(np.arange(num_classes))
plt.yticks(np.arange(num_classes))
plt.xlabel("Predicted label")
plt.ylabel("True label")
plt.title("Confusion Matrix")
plt.show()