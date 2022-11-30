import argparse

# Custom Package
from tqdm.auto import tqdm
from Model.Base_Model import ANN
from Train.Base_Model import *
from torch.utils.data import DataLoader

# Option
def parse_args():
    parser = argparse.ArgumentParser(description='BaseLine Model')

    # DataType
    parser.add_argument('--over_sampling', help='0:No, 1:SMOTE Oversampling', type=int, default=0)

    # Device
    parser.add_argument('--device', help='GPU Device', type=int, default=0)

    return parser.parse_args()

# Main
if __name__ == '__main__':
    # Option Setting
    args = parse_args()

    # Setting Seed
    device = torch.device('cuda:' + str(args.device)) if torch.cuda.is_available() else torch.device('cpu')

    CFG = {
        'EPOCHS': 1000,
        'BATCH_SIZE': 256,
        'SEED': 41,
        'In_Hidden': 27,
        'S_In_Hidden': 14,
        'Print_epoch': 50,
        'LEARNING_RATE': None,
        'REG': None,
        'In_hidden_list': None,
        'MIN_LR': None,
    }

    seed_everything(CFG['SEED'])

    # Data Load
    sampling_list = ['no', 'over']
    sampling = sampling_list[args.over_sampling]

    # Train
    train_dataset = CustomDataset(kind='train', sampling=sampling)
    train_loader = DataLoader(train_dataset, batch_size=CFG['BATCH_SIZE'], shuffle=True)

    # Validation
    val_dataset = CustomDataset(kind='val')
    val_loader = DataLoader(val_dataset, batch_size=CFG['BATCH_SIZE'], shuffle=True)

    # Test - Data
    test_x = np.load('/data/jyhwang/construction_machinery/Data/sampling/Test/x.npy')

    # Hyperparameter Search...
    hyper_parameters = Hyperparameters()
    hyperparameter_list = hyper_parameters.hyperparameter_list
    str_hyper_parameter_list = hyper_parameters.str_hyper_parameter_list

    for i, hyper_parameter in enumerate(tqdm(hyperparameter_list, desc='Hyperparameter Search...')):
        CFG['LEARNING_RATE'] = hyper_parameter['lr']
        CFG['MIN_LR'] = hyper_parameter['min_lr']
        CFG['REG'] = hyper_parameter['reg']
        CFG['In_hidden_list'] = hyper_parameter['dimension']

        # Train Teacher Model
        model = ANN([CFG['In_Hidden']] + CFG['In_hidden_list'])
        model.eval()
        optimizer = torch.optim.Adam(model.parameters(), lr=CFG['LEARNING_RATE'], weight_decay=CFG['REG'])
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=1,
                                                               threshold_mode='abs', min_lr=CFG['MIN_LR'],
                                                               verbose=False)
        teacher_model, T_F1 = train(model, optimizer, train_loader, val_loader, scheduler, device, CFG)

        # Student Model
        student_model = ANN([CFG['S_In_Hidden']] + CFG['In_hidden_list'])
        student_model.eval()
        optimizer = torch.optim.Adam(student_model.parameters(), lr=CFG['LEARNING_RATE'])
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=1,
                                                               threshold_mode='abs', min_lr=1e-6, verbose=False)
        best_student_model, S_F1 = student_train(student_model, teacher_model, optimizer, train_loader, val_loader, scheduler,
                                           device, CFG)

        # Check Threshold
        best_threshold, best_score = choose_threshold(best_student_model, val_loader, device)
        print(f'Best Threshold : [{best_threshold}], Score : [{best_score:.5f}]')

        # Save Result
        hyper_parameters.save_val_test({'T-F1': T_F1, 'S-F1': S_F1, 'Final': best_score},
                                       sampling+'Base',  str_hyper_parameter_list,
                                       str_hyper_parameter_list[i])

        # Test
        test_save_path = './Result/Submit/'+sampling+'Base'
        createFolder(test_save_path)
        test(best_student_model, test_x, best_threshold, test_save_path, device, str_hyper_parameter_list[i]+'.csv')