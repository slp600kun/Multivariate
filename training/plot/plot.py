import matplotlib.pyplot as plt

def plot_loss(file_paths, labels,plot_type,y_label,title,output_path):
    plt.figure()
    
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
            if plot_type == 'accuracy':
                plot_line = float(data[4].strip())
            steps.append(step)
            plot_lines.append(plot_line)
        
        plt.plot(steps, plot_lines, label=labels[i])
    
    plt.xlabel('Step')
    plt.ylabel(y_label)
    plt.title(title)
    plt.legend()
    plt.savefig(output_path, format='png')
    plt.show()

# ファイルパスとラベルのリスト
file_paths = ['data/LSTM_classfier1_train_accuracies.txt', 'data/LSTM_classfier1_val_accuracies.txt']
labels = ['train_loss', 'val_loss']
y_label = 'cross entropy loss'
title = 'LSTM-classifier1 loss'
output_path = "out/LSTM-classifier1_loss.png"
# プロットの実行
plot_loss(file_paths, labels,'loss',y_label,title,output_path)