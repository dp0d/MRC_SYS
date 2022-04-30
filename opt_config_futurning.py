'''base_line'''
#     expected_version = '2021'
#     parser = argparse.ArgumentParser(
#     description='Search hyperparameters' + expected_version)
#     parser.add_argument('--learning_rate', help='learning_rate')
#     parser.add_argument('--num_train_epochs', help='train_epochs')
#     parser.add_argument('--batch_size', help='batch_size')
#     _args = parser.parse_args()
#     import args
#     args.learning_rate = float(_args.learning_rate)
#     args.num_train_epochs = int(_args.num_train_epochs)
#     args.batch_size = int(_args.batch_size)
#     from train import train
#     train()  



import argparse 

if __name__ == '__main__':
 
    expected_version = '2021'
    parser = argparse.ArgumentParser(
    description='Search hyperparameters' + expected_version)
    parser.add_argument('--epsilon', help='epsilon')
    _args = parser.parse_args()
    import args
    args.epsilon = float(_args.epsilon)
    from train_adv_Copy import _train
    _train()
        
    