import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import mean_absolute_error

from Utils.metric_utils import extract_time


class PredictorModel(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(PredictorModel, self).__init__()
        self.gru = nn.GRU(input_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, 1)  
    def forward(self, x, seq_lengths):
        packed_input = torch.nn.utils.rnn.pack_padded_sequence(x, seq_lengths, batch_first=True, enforce_sorted=False)
        gru_out, _ = self.gru(packed_input)
        unpacked_out, _ = torch.nn.utils.rnn.pad_packed_sequence(gru_out, batch_first=True)
        output = self.fc(unpacked_out)
        return torch.sigmoid(output)

def predictive_score(ori_data, generated_data):
    """Report the performance of Post-hoc RNN one-step ahead prediction using PyTorch."""
    
    no, seq_len, dim = ori_data.shape
    
    ori_time, ori_max_seq_len = extract_time(ori_data)
    generated_time, generated_max_seq_len = extract_time(generated_data)
    hidden_dim = int(dim / 2)
    iterations = 5000
    batch_size = 128
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model = PredictorModel(input_dim=dim-1, hidden_dim=hidden_dim).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    for itt in range(iterations):
        idx = np.random.permutation(len(generated_data))
        train_idx = idx[:batch_size]     
            
        X_mb = [torch.tensor(generated_data[i][:-1, :(dim-1)], dtype=torch.float32).to(device) for i in train_idx]
        T_mb = [generated_time[i] - 1 for i in train_idx]
        Y_mb = [torch.tensor(np.reshape(generated_data[i][1:, (dim-1)], [len(generated_data[i][1:, (dim-1)]), 1]), dtype=torch.float32).to(device) for i in train_idx]
        
        X_mb = torch.nn.utils.rnn.pad_sequence(X_mb, batch_first=True, padding_value=0)
        Y_mb = torch.nn.utils.rnn.pad_sequence(Y_mb, batch_first=True, padding_value=0)
        
        optimizer.zero_grad()
        outputs = model(X_mb, torch.tensor(T_mb).to(device))  
        
    
        loss = criterion(outputs, Y_mb)
        loss.backward()
        optimizer.step()
    
    idx = np.random.permutation(len(ori_data))
    train_idx = idx[:no]

    X_mb = [torch.tensor(ori_data[i][:-1, :(dim-1)], dtype=torch.float32).to(device) for i in train_idx]
    T_mb = [ori_time[i] - 1 for i in train_idx]
    Y_mb = [torch.tensor(np.reshape(ori_data[i][1:, (dim-1)], [len(ori_data[i][1:, (dim-1)]), 1]), dtype=torch.float32).to(device) for i in train_idx]
    
    X_mb = torch.nn.utils.rnn.pad_sequence(X_mb, batch_first=True, padding_value=0)
    Y_mb = torch.nn.utils.rnn.pad_sequence(Y_mb, batch_first=True, padding_value=0)
    
    # Prediction
    with torch.no_grad():
        pred_Y_curr = model(X_mb, torch.tensor(T_mb).to(device))
    
    MAE_temp = 0
    for i in range(no):
        MAE_temp += mean_absolute_error(Y_mb[i], pred_Y_curr[i, :, :])
    
    predictive_score = MAE_temp / no
    
    return predictive_score


