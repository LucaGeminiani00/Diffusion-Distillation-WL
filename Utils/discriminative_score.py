import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score
from tqdm.auto import tqdm

from Utils.metric_utils import extract_time, train_test_divide


def batch_generator(data, time, batch_size):
    no = len(data)
    idx = np.random.permutation(no)
    train_idx = idx[:batch_size]

    X_mb = [data[i] for i in train_idx]
    T_mb = [time[i] for i in train_idx]

    return X_mb, T_mb

class DiscriminatorModel(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(DiscriminatorModel, self).__init__()
        self.gru = nn.GRU(input_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, 1) 
    
    def forward(self, x, seq_lengths):
        packed_input = torch.nn.utils.rnn.pack_padded_sequence(x, seq_lengths, batch_first=True, enforce_sorted=False)
        gru_out, _ = self.gru(packed_input)
        unpacked_out, _ = torch.nn.utils.rnn.pad_packed_sequence(gru_out, batch_first=True)
        output = self.fc(unpacked_out)
        return output  

def discriminative_score(ori_data, generated_data):
    
    no, seq_len, dim = np.asarray(ori_data).shape    
    
    ori_time, ori_max_seq_len = extract_time(ori_data)
    generated_time, generated_max_seq_len = extract_time(generated_data)
    
    hidden_dim = int(dim / 2)
    iterations = 2000
    batch_size = 128
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model = DiscriminatorModel(input_dim=dim, hidden_dim=hidden_dim).to(device)
    criterion = nn.BCEWithLogitsLoss()  
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    train_x, train_x_hat, test_x, test_x_hat, train_t, train_t_hat, test_t, test_t_hat = \
        train_test_divide(ori_data, generated_data, ori_time, generated_time)

    for itt in tqdm(range(iterations), desc='training', total=iterations):
          
        X_mb, T_mb = batch_generator(train_x, train_t, batch_size)
        X_hat_mb, T_hat_mb = batch_generator(train_x_hat, train_t_hat, batch_size)
       
        X_mb = torch.stack([torch.tensor(x, dtype=torch.float32).to(device) for x in X_mb])
        T_mb = torch.tensor(T_mb, dtype=torch.long).to(device)
        X_hat_mb = torch.stack([torch.tensor(x, dtype=torch.float32).to(device) for x in X_hat_mb])
        T_hat_mb = torch.tensor(T_hat_mb, dtype=torch.long).to(device)

        optimizer.zero_grad()
        y_logit_real = model(X_mb, T_mb.cpu())
        y_logit_fake = model(X_hat_mb, T_hat_mb.cpu())
        
        labels_real = torch.ones_like(y_logit_real)
        labels_fake = torch.zeros_like(y_logit_fake)
    
        d_loss_real = criterion(y_logit_real, labels_real)
        d_loss_fake = criterion(y_logit_fake, labels_fake)
        d_loss = d_loss_real + d_loss_fake
     
        d_loss.backward()
        optimizer.step()
     
    test_x = torch.stack([torch.tensor(x, dtype=torch.float32).to(device) for x in test_x])
    test_t = torch.tensor(test_t, dtype=torch.long).to(device)
    test_x_hat = torch.stack([torch.tensor(x, dtype=torch.float32).to(device) for x in test_x_hat])
    test_t_hat = torch.tensor(test_t_hat, dtype=torch.long).to(device)

    with torch.no_grad():
        y_pred_real_curr = torch.sigmoid(model(test_x, test_t.cpu()))
        y_pred_fake_curr = torch.sigmoid(model(test_x_hat, test_t_hat.cpu()))

    y_pred_real_curr_bin = (y_pred_real_curr > 0.5).int()  
    y_pred_fake_curr_bin = (y_pred_fake_curr > 0.5).int()  

    y_pred_real_curr_bin = y_pred_real_curr_bin.view(-1)  
    y_pred_fake_curr_bin = y_pred_fake_curr_bin.view(-1)  

    y_pred_final = torch.cat((y_pred_real_curr_bin, y_pred_fake_curr_bin), dim=0)
    y_label_final = torch.cat((torch.ones(y_pred_real_curr_bin.size(0)), torch.zeros(y_pred_fake_curr_bin.size(0))), dim=0)


    acc = accuracy_score(y_label_final.cpu(), y_pred_final.cpu())
    #fake_acc = accuracy_score(np.zeros(len(y_pred_fake_curr_bin)).cpu(), y_pred_fake_curr_bin.cpu())
    #real_acc = accuracy_score(np.ones(len(y_pred_real_curr_bin)).cpu(), y_pred_real_curr_bin.cpu())

    discriminative_score = np.abs(0.5 - acc)

    return discriminative_score