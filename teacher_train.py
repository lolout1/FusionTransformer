from tqdm import tqdm
from arguments import parse_args
from teacherNet import teacherNet
import torch.nn as nn 
import torch.nn.functional as F
import torch.optim as optim 
from torchvision import datasets, transforms
import os 
import torch
import time



start_time = time.time()

args = parse_args()

args.cuda = not args.no_cuda and torch.cuda.is_available()

torch.manual_seed(args.seed)

if args.cuda: 
    torch.cuda.manual_seed(args.seed)


kwargs = {'num_workers' : 0, 'pin_memory' : True} if args.cuda else {}

mnist_training_set = datasets.MNIST('./data_mnist', train = True, download = True,
                                transform =  transforms.Compose([
                                    transforms.ToTensor(), 
                                    transforms.Normalize((0.1307,),(0.3081,))
                                ]))

#Split the training set into training and validation sets
train_size = int(0.8 * len(mnist_training_set))
val_size = len(mnist_training_set) -train_size
train_dataset , val_dataset = torch.utils.data.random_split(mnist_training_set,[train_size , val_size])

train_loader = torch.utils.data.DataLoader(
    train_dataset, batch_size = args.batch_size, shuffle = True, **kwargs
)

val_loader = torch.utils.data.DataLoader(
    val_dataset, batch_size = args.batch_size, shuffle = True, **kwargs
)




def train(epoch,num_epochs,best_accuracy, model):
    model.train()
    with tqdm(total = len(train_loader), desc = f'Epoch {epoch}/{num_epochs}' , ncols = 128) as pbar :
        for batch_idx, (data,target) in enumerate(train_loader):
            if args.cuda:
                data, target = data.cuda(), target.cuda()
            

            # data= torch.tensor(data)
            # target = torch.tensor(target)
            optimizer.zero_grad()
            output = model(data)
            loss = F.cross_entropy(output, target)
            loss.backward()
            optimizer.step()

            pbar.update(1)
        
    
    #Evaluate the model on the validation data
    model.eval()
    
    with torch.no_grad():
            correct = 0 
            total = 0
            train_loss = 0
            for data , target  in val_loader:
                if args.cuda:
                    data, target = data.cuda(), target.cuda()
                
            # data = torch.tensor(data)
            # target = torch.tensor(target)
            output = model(data)
            _, predicted = torch.max(output.data, 1)
            total += target.size(0)
        
            correct += (predicted == target).sum().item()
            val_acc = correct/total


            if best_accuracy < val_acc :
                best_accuracy = val_acc
                # torch.save(model.state_dict(), 'best_ckpt.pt')
                # print('Checkpoint Saved')
            print('Epoch [{}/{}], Loss: {:.4f}, Val Acc: {:.2f}%'.format(
            epoch+1, num_epochs, loss.item(), val_acc*100
        ))


if __name__ == "__main__":
    best_accuracy = 0 
    model = teacherNet()
    optimizer = optim.SGD(model.parameters(), lr = args.lr, momentum = args.momentum, 
                      weight_decay = 5e-4)
    if args.cuda:
        model.cuda()
    for epoch in range(args.epochs):
        train(epoch, args.epochs, best_accuracy, model)

        




