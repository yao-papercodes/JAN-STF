import pickle

dataset = ['Houston13', 'Dioni', 'paviaU']
prepath = '.'
log_name = '2023-10-02-09:41:03'
dataset_no = 2
text_to_copy = ''
with open(f'{prepath}/exp/{log_name}/train_times_{dataset[dataset_no]}.pickle', 'rb') as f:
    data = pickle.load(f)


dataset = ['Houston13', 'Dioni', 'paviaU']
prepath = '.'
log_name = '2023-10-03-00:35:08'
dataset_no = 2
text_to_copy = ''
with open(f'{prepath}/exp/{log_name}/train_times_{dataset[dataset_no]}.pickle', 'rb') as f:
    data2 = pickle.load(f)


#@ acc_test_list
data['acc_test_list'][-5:] = data2['acc_test_list'][:]
#@ kappa_test_list
data['kappa_test_list'][-2:] = data2['kappa_test_list'][:]
#@ acc_class_test_list
data['acc_class_test_list'][-2:] = data2['acc_class_test_list'][:]


#! save
with open(f'./train_times_{dataset[dataset_no]}.pickle', 'wb') as f:
    pickle.dump(data, f)