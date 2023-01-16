import matplotlib.pyplot as plt
import numpy as np
import os
path = 'data_A'

files = os.listdir(path)

for i in range(len(files)):
    file = path + '/' + files[i]
    a = np.loadtxt(file)
    x = a[:, 0]
    y = a[:, 1]
    plt.plot(x, y, label=files[i][:-4])

