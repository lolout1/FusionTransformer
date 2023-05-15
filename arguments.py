import argparse

def parse_args():

    parser = argparse.ArgumentParser(description = 'Distillation')
    parser.add_argument('--batch-size', type = int, default = 8, metavar = 'N',
                        help = 'input batch size for training (default: 8)')

    parser.add_argument('--test-batch-size', type = int, default = 8, 
                        metavar = 'N', help = 'input batch size for testing(default: 1000)')

    parser.add_argument('--epochs', type = int , default = 70, metavar = 'N', 
                        help = 'number of epochs to train (default: 10)')

    parser.add_argument('--lr', type = float, default = 0.0025, metavar = 'LR', 
                        help = 'learning rate (default: 0.01)')

    parser.add_argument('--momentum', type = float, default = 0.9, metavar = 'M',
                        help = 'SGD momentum (default: 0.5)')

    parser.add_argument('--no-cuda', action = 'store_true', default = False, 
                        help = 'disables CUDA training')
    
    parser.add_argument('--seed', type =  int , default = 1 , help = 'random seed (default: 1)') 

    parser.add_argument('--log-interval', type = int , default = 10, metavar = 'N',
                        help = 'how many bathces to wait before logging training status')
    parser.add_argument('--val-bsize', type = int, default = 8, metavar = 'N', 
                        help = 'Validation batch sieze (default : 8)')
    parser.add_argument('--dataset', type = str, default= 'ncrc', metavar = 'D', help = 'Which dataset to use')

    parser.add_argument('--mocap', type = int, default = 600, help = 'Skeleton Frame number')

    parser.add_argument('--acc-frame', type = int , default = 150, help = 'Acceleton frame number')

    parser.add_argument('--num-joints', type = int, default = 29, help = 'Number of joints in skeleton data')

    parser.add_argument('--num-classes', type = int, default = 6, help = 'Number of classes in data')

    parser.add_argument('--fusion', type = str, default = 'simple', metavar = 'F', help = "Fusion method to choose (default : Simple)")

    arg = parser.parse_args()

    return arg

if __name__ == "__main__":
    args  = parse_args() 
    print(args)