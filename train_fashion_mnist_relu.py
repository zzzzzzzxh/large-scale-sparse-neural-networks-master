import argparse
import logging
import tensorflow as tf
from mpi4py import MPI
from time import time
from utils.load_data import *
from utils.result_utils import *
from utils.nn_functions import AlternatedLeftReLU, Softmax, Relu, Sigmoid
from wasap_sgd.mpi.manager import MPIManager
from wasap_sgd.train.algo import Algo
from wasap_sgd.train.data import Data
from torch.utils.data import DataLoader
from wasap_sgd.train.model import SETMPIModel
from wasap_sgd.logger import initialize_logger
from sklearn.metrics import confusion_matrix
from sklearn.metrics import cohen_kappa_score
import matplotlib.pyplot as plt


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--monitor', help='Monitor cpu and gpu utilization', default=False, action='store_true')

    # Configuration of network topology
    parser.add_argument('--masters', help='number of master processes', default=1, type=int)
    parser.add_argument('--processes', help='number of processes per worker', default=1, type=int)
    parser.add_argument('--synchronous', help='run in synchronous mode', action='store_true')

    # Configuration of training process
    parser.add_argument('--loss', help='loss function', default='cross_entropy')
    parser.add_argument('--sync-every', help='how often to sync weights with master',
                        default=1, type=int, dest='sync_every')
    parser.add_argument('--mode', help='Mode of operation.'
                        'One of "sgd" (Stohastic Gradient Descent), "sgdm" (Stohastic Gradient Descent with Momentum)',
                        default='sgdm')

    # logging configuration
    parser.add_argument('--log-file', default=None, dest='log_file',
                        help='log file to write, in additon to output stream')
    parser.add_argument('--log-level', default='info', dest='log_level', help='log level (debug, info, warn, error)')

    # Model configuration
    parser.add_argument('--batch-size', type=int, default=128, help='input batch size for training (default: 128)')
    parser.add_argument('--epochs', type=int, default=10,  help='number of epochs to train (default: 10)')
    parser.add_argument('--lr', type=float, default=0.01, help='learning rate (default: 0.01)')
    parser.add_argument('--lr-rate-decay', type=float, default=0.0, help='learning rate decay (default: 0)')
    parser.add_argument('--momentum', type=float, default=0.9, help='SGD momentum (default: 0.9)')
    parser.add_argument('--dropout-rate', type=float, default=0.3, help='Dropout rate (default: 0.3)')
    parser.add_argument('--weight-decay', type=float, default=0.0, help='Weight decay (l2 regularization)')
    parser.add_argument('--epsilon', type=int, default=20, help='Sparsity level (default: 20)')
    parser.add_argument('--zeta', type=float, default=0.3,
                        help='It gives the percentage of unimportant connections which are removed and replaced with '
                             'random ones after every epoch(in [0..1])')
    parser.add_argument('--n-neurons', type=int, default=3000, help='Number of neurons in the hidden layer')
    parser.add_argument('--prune', default=True, help='Perform Importance Pruning', action='store_true')
    parser.add_argument('--seed', type=int, default=1, help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=10,
                        help='how many batches to wait before logging training status')
    parser.add_argument('--n-training-samples', type=int, default=60000, help='Number of training samples')
    parser.add_argument('--n-testing-samples', type=int, default=10000, help='Number of testing samples')
    parser.add_argument('--augmentation', default=True, help='Data augmentation', action='store_true')
    parser.add_argument('--dataset', default='fashionmnist', help='Specify dataset. One of "cifar10", "fashionmnist",'
                                                             '"madalon",  or "mnist"')

    args = parser.parse_args()

    if args.dataset == 'fashionmnist' or args.dataset == 'mnist':
        # Model architecture mnist
        dimensions = (784, 1000, 1000, 1000, 10)
        loss = 'cross_entropy'
        weight_init = 'he_uniform'
        # activations = (AlternatedLeftReLU(0.6), AlternatedLeftReLU(0.6), AlternatedLeftReLU(0.6), Softmax)
        activations = (Relu, Relu, Relu, Softmax)
        if args.dataset == 'fashionmnist':
            X_train, Y_train, X_test, Y_test = load_fashion_mnist_data(args.n_training_samples, args.n_testing_samples)
        else:
            X_train, Y_train, X_test, Y_test = load_mnist_data(args.n_training_samples, args.n_testing_samples)
    elif args.dataset == 'madalon':
        # Model architecture madalon
        dimensions = (500, 400, 100, 400, 1)
        loss = 'mse'
        activations = (Relu, Relu, Relu, Sigmoid)
        X_train, Y_train, X_test, Y_test = load_madelon_data()
    elif args.dataset == 'cifar10':
        # Model architecture cifar10
        dimensions = (3072, 4000, 1000, 4000, 10)
        weight_init = 'he_uniform'
        loss = 'cross_entropy'
        activations = (AlternatedLeftReLU(-0.75), AlternatedLeftReLU(0.75), AlternatedLeftReLU(-0.75), Softmax)
        if args.augmentation:
            X_train, Y_train, X_test, Y_test = load_cifar10_data_not_flattened(args.n_training_samples, args.n_testing_samples)
        else:
            X_train, Y_train, X_test, Y_test = load_cifar10_data(args.n_training_samples, args.n_testing_samples)
    else:
        raise NotImplementedError("The given dataset is not available")

    weight_init = 'he_uniform'
    prune = args.prune
    n_hidden_neurons = args.n_neurons
    epsilon = args.epsilon
    zeta = args.zeta
    n_epochs = 500
    batch_size = 128
    dropout_rate = args.dropout_rate
    learning_rate = args.lr
    momentum = args.momentum
    weight_decay = args.weight_decay
    n_training_samples = args.n_training_samples
    n_testing_samples = args.n_testing_samples
    learning_rate_decay = args.lr_rate_decay
    class_weights = None
    num_workers = 0

    # SET parameters
    model_config = {
        'n_epochs': n_epochs,
        'batch_size': batch_size,
        'dropout_rate': dropout_rate,
        'lr': learning_rate,
        'zeta': zeta,
        'epsilon': epsilon,
        'momentum': momentum,
        'weight_decay': 0.0,
        'n_hidden_neurons': n_hidden_neurons,
        'n_training_samples': n_training_samples,
        'n_testing_samples': n_testing_samples,
        'loss': loss,
        'weight_init': weight_init,
        'prune': prune,
        'num_workers': num_workers
    }
    model = SETMPIModel(dimensions, activations, class_weights=class_weights, **model_config)

    validate_every = int(X_train.shape[0] // (batch_size * args.sync_every))
    data = Data(batch_size=batch_size,
                x_train=X_train, y_train=Y_train,
                x_test=X_test, y_test=Y_test, augmentation=True,
                dataset=args.dataset)
    del X_train, Y_train, X_test, Y_test
    data.shuffle()
    train_generate = data.generate_train_data()
    test_generate = data.generate_test_data()
    x_test = data.get_test_data()
    y_test = data.get_test_labels()



    #Train the model
    results = {'acc': [], 'kappa': [], 'EGF': [], 'overfitting': []}

    for epoch in range(n_epochs):
        da = {}
        EGF = 0
        for j in range(data.x_train.shape[0] // data.batch_size):
            start_pos = j * data.batch_size
            end_pos = (j + 1) * data.batch_size
            x_train, y_train = data.x_train[start_pos:end_pos], data.y_train[start_pos:end_pos]
            update = model.train_on_batch(x_train, y_train)
            # update new activation function
            # for i in range(2,5):
            #     index = i
            #     slope = model.activations[index].slope
            #
            #     dw = update[index][0]
            #     delta = update[index][1]
            #     if index not in da:
            #         da[index] = - model.learning_rate * (np.mean(dw, axis=0).mean() + delta.mean())
            #     else:
            #         da[index] = model.momentum * da[index] - model.learning_rate * (np.mean(dw,axis=0).mean()+delta.mean())
            #
            #     slope += da[index]
            #     model.activations[index] = AlternatedLeftReLU(slope)


            model.apply_update(update)
            #计算EGF
            for i in range(1,5):
                EGF += sum(list(map(abs,update[i][0].data.tolist())))
        # print(model.activations[2].slope,model.activations[3].slope,model.activations[4].slope)
        EGF = EGF/(data.x_train.shape[0] * (model.n_layers-1))


        acc, y_pre = model.predict(x_test, y_test)
        acc_train, _ = model.predict(data.x_train, data.y_train)
        overfitting = acc_train - acc
        #compute kappa
        y_pre = model.to_label(y_pre)
        y_pre_label = np.where(y_pre == 1)[1].tolist()
        y_test_label = np.where(y_test == 1)[1].tolist()
        #get confusion_matrix

        C = confusion_matrix(y_test_label, y_pre_label)
        #画出混淆矩阵
        # show_confusion_matrix(C)
        #计算kappa系数
        kappa = cohen_kappa_score(y_test_label, y_pre_label)

        print("epoch : ", epoch, "acc : ", acc, 'kappa : ', kappa, 'EGF : ', EGF)
        results['acc'].append(acc)
        results['kappa'].append(kappa)
        results['EGF'].append(EGF)
        results['overfitting'].append(overfitting)
        if (epoch < n_epochs - 1):  # do not change connectivity pattern after the last epoch
            model.weight_evolution(epoch)

    #保存结果
    save_dic('result/FasionMNIST_relu_500epoch_epsilon20_3.txt', results)
    #读取结果
    # results = read_dic('result/temp.txt')







    #开始分析，loss，acc，kappa，EGF






# divide into two part

# docs this in theise
#
#
# show gradient flow

