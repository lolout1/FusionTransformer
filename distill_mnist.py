import torch 
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable 
from teacherNet import teacherNet
from student import studentNet
from tqdm import tqdm
import os 
import time
from arguments import parse_args
 
start_time = time.time()

#Traning settings 

args = parse_args()

args.cuda = not args.no_cuda and torch.cuda.is_available()

torch.manual_seed(args.seed)

if args.cuda:
    torch.cuda.manual_seed(args.seed) # set seed for random number generation 


kwargs = {'num_workers' : 0, 'pin_memory' : True } if args.cuda else {}

mnist_training_set = datasets.MNIST('./data_mnist', train = True, download = True,
                                transform =  transforms.Compose([
                                    transforms.ToTensor(), 
                                    transforms.Normalize((0.1307,),(0.3081,))
                                ]))

#Split the training set into training and validation sets
train_size = int(0.8 * len(mnist_training_set))
val_size = len(mnist_training_set) - train_size
train_dataset , val_dataset = torch.utils.data.random_split(mnist_training_set,[train_size, val_size])

train_loader = torch.utils.data.DataLoader(
                    train_dataset, batch_size = args.batch_size, shuffle = True,  **kwargs
                    )
val_loader = torch.utils.data.DataLoader(
    val_dataset, batch_size = args.val_bsize, shuffle = True , **kwargs
)                    



student_model = studentNet()
if args.cuda : 
    student_model.cuda()


optimizer = optim.SGD(student_model.parameters(), lr = args.lr, momentum = args.momentum)

def distillation(y, labels, teacher_scores, T, alpha):
    # Implementing alpha * Temp ^2 * crossEn(Q_s, Q_t) + (1-alpha)* crossEn(Q_s, y_true)
    pred_soft = F.log_softmax(y/T, dim = 1)
    # print(f'Student pred has Nan : {torch.isnan(pred_soft).any()}')
    teacher_scores_soft = F.log_softmax(teacher_scores/T, dim = 1)
    # print(f'Teacher pred has Nan : {torch.isnan(teacher_scores_soft).any()}')
    kl_div = nn.KLDivLoss(reduction= "batchmean", log_target=True)(pred_soft, teacher_scores_soft) * ( alpha * T * T * 2.0)
    # print(f'KlDiv pred has Nan : {torch.isnan(kl_div).any()}')
    loss_y_label = F.cross_entropy(y, labels) * (1.0 - alpha)
    # print(f'Y loss has Nan : {torch.isnan(loss_y_label).any()}')
    return kl_div + loss_y_label


def train(epoch,num_epochs, best_accuracy, student_model, teacher_model, loss_fn):
    student_model.train()
    teacher_model.eval()
    with tqdm(total = len(train_loader), desc = f'Epoch {epoch+1}/{num_epochs}', ncols = 128) as pbar:
        for batch_idx, (data, target) in enumerate(train_loader):
            if args.cuda:
                data, target = data.cuda(), target.cuda()
            optimizer.zero_grad()
            output = student_model(data)

            teacher_output = teacher_model(data)
            teacher_output = teacher_output.detach()
            loss = loss_fn(output, target, teacher_output, T = 2.0, alpha = 0.7)
            # if epoch == 1 : 
            #     print(f'Output: {batch_idx}\n')
            #     print(output)
            #     print(f'Loss: {batch_idx}\n')
            #     print(loss)
            loss.backward()
            optimizer.step()
            pbar.update(1)

    student_model.eval()

    with torch.no_grad():
        correct = 0
        total = 0
        for data, target in val_loader:
            if args.cuda:
                data,target = data.cuda(), target.cuda()
        output = student_model(data)
        _, predicted = torch.max(output.data, 1)
        total += target.size(0)
        correct += (predicted == target).sum().item()
        val_acc = correct / total

        if best_accuracy < val_acc: 
            best_accuracy = val_acc
        
        
        print('Epoch [{}]/[{}] , Loss: {:4f}, Val Acc: {:.2f}%'.format(
            epoch + 1, num_epochs, loss.item(), val_acc* 100
        ))




if __name__ == "__main__":
    best_accuracy = 0
    teacher_model = teacherNet()
    teacher_model.load_state_dict(torch.load('best_ckpt.pt'))

    for epoch in range(1, args.epochs + 1):
        train(epoch, num_epochs=args.epochs,
         student_model= student_model,loss_fn=distillation,best_accuracy=best_accuracy,
          teacher_model=teacher_model )