import matplotlib.pyplot as plt
import numpy as np
import os
from utils.result_utils import *


def PlotActiFunc(x, y, title):
    plt.grid(which='minor', alpha=0.2)
    plt.grid(which='major', alpha=0.5)
    plt.plot(x, y)
    plt.title(title)
    plt.show()


def PlotMultiFunc(x, y):
    plt.grid(which='minor', alpha=0.2)
    plt.grid(which='major', alpha=0.5)
    plt.plot(x, y)


if __name__ == '__main__':
    x = np.arange(0, 500, 1)
    # activateFunc = ActivateFunc(x)
    # activateFunc.b = 1
    # activateFunc.a = 100
    # PlotActiFunc(x, activateFunc.LeakyReLU()[0], title='LeakyReLU')
    # PlotActiFunc(x, activateFunc.ReLU()[0], title='ReLU')
    # PlotActiFunc(x, activateFunc.PReLU()[0], title='PReLU')
    #
    # PlotMultiFunc(x, activateFunc.Mish()[1])
    # PlotMultiFunc(x, activateFunc.Swish()[1])
    # plt.legend(['Mish-grad', 'Swish-grad'])
    # plt.figure(1)


    Lrelu_res = read_dic('result/FasionMNIST_Lrelu_0.6_500epoch_epsilon20.txt')
    relu_res = read_dic('result/FasionMNIST_relu_500epoch_epsilon20.txt')
    allrelu_res = read_dic('result/FasionMNIST_allrelu_0.6_500epoch_epsilon20.txt')
    grelu_res = read_dic('result/FasionMNIST_grelu_0.6_500epoch_epsilon20_new.txt')

    lrelu_arr = [max(read_dic('result/madalon_Lrelu_0.5_500epoch_epsilon10.txt')['acc']),
                 max(read_dic('result/madalon_Lrelu_0.5_500epoch_epsilon10_2.txt')['acc']),
                 max(read_dic('result/madalon_Lrelu_0.5_500epoch_epsilon10_3.txt')['acc'])]
    lrelu_arr_mean = np.mean(lrelu_arr)*100
    lrelu_arr_std = np.std(lrelu_arr)*100

    relu_arr = [max(read_dic('result/madalon_relu_500epoch_epsilon10.txt')['acc']),
                 max(read_dic('result/madalon_relu_500epoch_epsilon10_2.txt')['acc']),
                 max(read_dic('result/madalon_relu_500epoch_epsilon10_3.txt')['acc'])]
    relu_arr_mean = np.mean(relu_arr)*100
    relu_arr_std = np.std(relu_arr)*100

    allrelu_arr = [max(read_dic('result/madalon_allrelu_0.5_500epoch_epsilon10.txt')['acc']),
                   max(read_dic('result/madalon_allrelu_0.5_500epoch_epsilon10_2.txt')['acc']),
                   max(read_dic('result/madalon_allrelu_0.5_500epoch_epsilon10_3.txt')['acc'])]
    allrelu_arr_mean = np.mean(allrelu_arr)*100
    allrelu_arr_std = np.std(allrelu_arr)*100

    grelu_arr = [max(read_dic('result/madalon_grelu_0.5_500epoch_epsilon10.txt')['acc']),
                 max(read_dic('result/madalon_grelu_0.5_500epoch_epsilon10_2.txt')['acc']),
                 max(read_dic('result/madalon_grelu_0.5_500epoch_epsilon10_3.txt')['acc'])]
    grelu_arr_mean = np.mean(grelu_arr)*100
    grelu_arr_std = np.std(grelu_arr)*100

    print("%.2f" % grelu_arr_mean,"%.2f" % grelu_arr_std)
    print("%.2f" % allrelu_arr_mean,"%.2f" % allrelu_arr_std)
    print("%.2f" % relu_arr_mean,"%.2f" % relu_arr_std)
    print("%.2f" % lrelu_arr_mean,"%.2f" % lrelu_arr_std)














    Lrelu_acc = Lrelu_res['acc']
    Lrelu_acc = [elem * 100 for elem in Lrelu_acc]
    Lrelu_kappa = Lrelu_res['kappa']
    Lrelu_EGF = Lrelu_res['EGF']
    Lrelu_ov = Lrelu_res['overfitting']

    relu_acc = relu_res['acc']
    relu_acc = [elem * 100 for elem in relu_acc]
    relu_kappa = relu_res['kappa']
    relu_EGF = relu_res['EGF']
    relu_ov = relu_res['overfitting']

    allrelu_acc = allrelu_res['acc']
    allrelu_acc = [elem * 100 for elem in allrelu_acc]
    allrelu_kappa = allrelu_res['kappa']
    allrelu_EGF = allrelu_res['EGF']
    allrelu_ov = allrelu_res['overfitting']

    grelu_acc = grelu_res['acc']
    grelu_acc = [elem * 100 for elem in grelu_acc]
    grelu_kappa = grelu_res['kappa']
    grelu_EGF = grelu_res['EGF']
    grelu_ov = grelu_res['overfitting']



    PlotMultiFunc(x, Lrelu_acc)
    PlotMultiFunc(x, relu_acc)
    PlotMultiFunc(x, allrelu_acc)
    PlotMultiFunc(x, grelu_acc)
    # plt.legend(['ReLU', 'All-ReLU', 'GAReLU'])
    plt.legend(['Leaky ReLU', 'ReLU', 'All-ReLU', 'GAReLU'])
    plt.xlabel("Epochs[#]")
    plt.ylabel("Accuracy on the test set[%]")
    plt.title('Performance on Fashion MNIST')
    # plt.ylim([50, 95])
    plt.figure(1)
    plt.show()

    PlotMultiFunc(x, Lrelu_EGF)
    PlotMultiFunc(x, relu_EGF)
    PlotMultiFunc(x, allrelu_EGF)
    PlotMultiFunc(x, grelu_EGF)
    # plt.legend(['ReLU', 'All-ReLU', 'GAReLU'])
    plt.legend(['Leaky ReLU', 'ReLU', 'All-ReLU', 'GAReLU'])
    plt.xlabel("Epochs[#]")
    plt.ylabel("Effective Gradient Flow")
    plt.title('Effective Gradient Flow on Fashion MNIST')
    # plt.ylim([50, 90])
    plt.figure(2)
    plt.show()


    PlotMultiFunc(x, Lrelu_ov)
    PlotMultiFunc(x, relu_ov)
    PlotMultiFunc(x, allrelu_ov)
    PlotMultiFunc(x, grelu_ov)
    # plt.legend(['ReLU', 'All-ReLU', 'GAReLU'])
    plt.legend(['Leaky ReLU', 'ReLU', 'All-ReLU', 'GAReLU'])
    plt.xlabel("Epochs[#]")
    plt.ylabel("Overfitting")
    plt.title('Overfitting on Fashion MNIST')
    # plt.ylim([50, 90])
    plt.figure(3)
    plt.show()

    # PlotActiFunc(x, Lrelu_kappa, title='kappa')
    # PlotActiFunc(x, Lrelu_EGF, title='EGF')
    # plt.figure(1)
