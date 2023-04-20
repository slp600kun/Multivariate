import torch
from train import SVM_for_two_dim
from torch.utils.data import DataLoader,TensorDataset


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#SVMマシンのモデル定義
svm_model = SVM_for_two_dim().to(device)
svm_loss_fn = torch.nn.HingeEmbeddingLoss().to(device)
svm_optimizer = torch.optim.SGD(svm_model.parameters(), lr=0.01)

#識別学習(svm)

#テンソルに変換
vector_train = torch.stack(feature_vector_train)
vector_val = torch.stack(feature_vector_val)
label_train = torch.tensor(feature_label_train)
label_val = torch.tensor(feature_label_val)

# データセットを作成
svm_train_dataset = TensorDataset(vector_train, label_train)
svm_val_dataset = TensorDataset(vector_val, label_val)
# データローダーを作成
svm_train_loader = DataLoader(svm_train_dataset,  batch_size= 128,shuffle=True)
svm_val_loader = DataLoader(svm_val_dataset, batch_size= 128,shuffle=True)

svm_model.train()
for epoch in range(100):
    
    svm_steps_losses = []
    svm_steps_accu = []

    for steps, (inputs, labels) in enumerate(svm_train_loader):
        outputs = svm_model(inputs.to(device))
        svm_loss = svm_loss_fn(outputs, labels.to(device))
        svm_steps_losses.append(svm_loss.cpu().detach().numpy())
        svm_loss.backward()
        svm_optimizer.step()
    print(f'Epoch {epoch}, train loss: {np.mean(svm_steps_losses)}, ')

    svm_model.eval()
    with torch.no_grad():
        correct_eval = 0
        total_eval = 0
        for eval_vec, eval_labels in svm_val_loader:
            output_eval = svm_model(eval_vec.to(device))
            predicted_eval = torch.sign(output_eval).squeeze().long()
            total_eval += eval_labels.to(device).size(0)
            correct_eval += (predicted_eval == eval_labels.to(device)).sum().item()
        accuracy_eval = correct_eval / total_eval

    print(f'Epoch {epoch}, Accuracy on evaluation data: {accuracy_eval}')
    file3.write("%s , %s, %s, %s, %s, %s\n" % (str(epoch), "loss", str(np.mean(svm_steps_losses)), "val_accuracy", str(accuracy_eval), str(now_time)))
file3.close()
