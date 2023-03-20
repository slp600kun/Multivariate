import matplotlib.pyplot as plt
import pandas as pd

# データの読み込み
f = open("training_accuracies.txt", "r")
hoge = pd.read_table(f,header=None,sep=",")
f.close()
print(hoge)

plt.figure()
hoge[2].plot()

plt.savefig("plot.png")
plt.close('all')