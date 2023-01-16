from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt


def show_confusion_matrix(C):
    plt.matshow(C, cmap=plt.cm.Reds)  # 根据最下面的图按自己需求更改颜色
    # plt.colorbar()

    for i in range(len(C)):
        for j in range(len(C)):
            plt.annotate(C[j, i], xy=(i, j), horizontalalignment='center', verticalalignment='center')

    # plt.tick_params(labelsize=15) # 设置左边和上面的label类别如0,1,2,3,4的字体大小。

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()


def save_dic(path, dic):
    f = open(path, 'w')
    f.write(str(dic))
    f.close()

def read_dic(path):
    # 读取
    f = open(path, 'r')
    a = f.read()
    dict_name = eval(a)
    f.close()
    return dict_name