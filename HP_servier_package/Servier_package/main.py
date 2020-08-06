import argparse
from Servier_package.codes.model1 import train_model1, evaluate_model1, predict_model1
from Servier_package.codes.model2 import train_model2, evaluate_model2, predict_model2
from Servier_package.codes.model3 import train_model3, evaluate_model3, predict_model3


def main():
    
    parser = argparse.ArgumentParser(description='Servier_tests')
    parser.add_argument('task', default='train',type=str, help='can be train, evaluate or predict')
    parser.add_argument('model_name', default='model1',type=str, help='can be model1, model2 or model3')
    parser.add_argument('--data_path',default='../datasets/dataset_single.csv',type=str, help='path to the .csv file')
    parser.add_argument('--lr',default=0.00001,help='learning rate')
    parser.add_argument('--batch_size',default=32,help='batch size')
    parser.add_argument('--num_epochs', default=300, help='number of training epochs')
    parser.add_argument('--smile',type=str,default='',help='enter smile to predict the class')
    parser.add_argument('--embedding_length',type=int,default=32,help='enter embedding length')
    parser.add_argument('--output_dir',type=str,default='model')
    args = parser.parse_args()
    print("arguments:",args)
    
    if args.task=='train':
        if args.model_name=='model1':
            train_model1(args)
            # evaluate_model1(args)
        elif args.model_name=='model2':
            train_model2(args)
            # evaluate_model2(args)
        elif args.model_name=='model3':
            train_model3(args)
            # evaluate_model3(args)
        else:
            print('model_name has to be either model1, model2 or model3.')
    elif args.task=='evaluate':
        if args.model_name=='model1':
            evaluate_model1(args)
        elif args.model_name=='model2':
            evaluate_model2(args)
        elif args.model_name=='model3':
            evaluate_model3(args)
        else:
            print('model_name has to be either model1, model2 or model3.')
    elif args.task=='predict':
        if args.smile == '':
            print("Please try again and enter smile in the command to be predicted.")
            exit()
        if args.model_name=='model1':
            predict_model1(args)
        elif args.model_name=='model2':
            predict_model2(args)
        elif args.model_name=='model3':
            predict_model3(args)
        else:
            print('Choose a task from train, evaluate, and predict.')
        

if __name__=="__main__":  
    main()
