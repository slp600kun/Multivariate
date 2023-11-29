import matplotlib.pyplot as plt

def plot_loss(file_paths, labels,plot_type,y_label,output_path):
    plt.figure()
    plt.rcParams["font.size"] = 15
    
    for i, file_path in enumerate(file_paths):
        steps = []
        plot_lines = []
        
        with open(file_path, 'r') as file:
            lines = file.readlines()
        
        for line in lines:
            data = line.strip().split(',')
            step = int(data[0].strip())
            if plot_type == 'loss':
                plot_line = float(data[2].strip())
                ax = plt.add_subplot(111)
                ax.set_xlim(0, 2)
                ax.set_ylim(-1, 2)
            if plot_type == 'accuracy':
                plot_line = float(data[4].strip())
            steps.append(step)
            plot_lines.append(plot_line)
        
        plt.plot(steps, plot_lines, label=labels[i])
    
    plt.xlabel('Step')
    plt.ylabel(y_label)
    plt.legend()
    plt.savefig(output_path, format='eps')
    plt.show()

# ファイルパスとラベルのリスト
file_paths = ['data/train_LSTM-FC.txt', 'data/val_LSTM-FC.txt']
labels = ['train_accuracy', 'val_accuracy']
y_label = 'accuracy'
output_path = "out/LSTM-FC_accuracy.eps"
# プロットの実行
plot_loss(file_paths, labels,'accuracy',y_label,output_path)