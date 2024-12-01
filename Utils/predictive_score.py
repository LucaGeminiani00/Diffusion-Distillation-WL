import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import mean_absolute_error
from tqdm.auto import tqdm

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
    
class PredictiveScoreModel():
    def __init__(self, ori_data, generated_data, iterations=5000, batch_size=128, lr=0.001, device=None):
        
        self.ori_data = ori_data
        self.generated_data = generated_data
        self.iterations = iterations
        self.batch_size = batch_size
        self.lr = lr
        self.device = device if device else torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.no, self.seq_len, self.dim = ori_data.shape
        self.ori_time, self.ori_max_seq_len = extract_time(self.ori_data)
        self.generated_time, self.generated_max_seq_len = extract_time(self.generated_data)
        
        self.hidden_dim = int(self.dim / 2)

        self.model = PredictorModel(input_dim=self.dim-1, hidden_dim=self.hidden_dim).to(self.device)
        self.criterion = nn.MSELoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)

    def train(self):
        
        for itt in tqdm(range(self.iterations), desc='Training Predictor', total=self.iterations):
            idx = np.random.permutation(len(self.generated_data))
            train_idx = idx[:self.batch_size]     
            
            X_mb = [torch.tensor(self.generated_data[i][:-1, :(self.dim-1)], dtype=torch.float32).to(self.device) for i in train_idx]
            T_mb = [self.generated_time[i] - 1 for i in train_idx]
            Y_mb = [torch.tensor(np.reshape(self.generated_data[i][1:, (self.dim-1)], [len(self.generated_data[i][1:, (self.dim-1)]), 1]), dtype=torch.float32).to(self.device) for i in train_idx]
            
            X_mb = torch.nn.utils.rnn.pad_sequence(X_mb, batch_first=True, padding_value=0)
            Y_mb = torch.nn.utils.rnn.pad_sequence(Y_mb, batch_first=True, padding_value=0)
            
            self.optimizer.zero_grad()
            outputs = self.model(X_mb, torch.tensor(T_mb).cpu())  
        
            loss = self.criterion(outputs, Y_mb)
            loss.backward()
            self.optimizer.step()

    def compute_pred(self, fake_data):
        """Compute the predictive score using ori_data and the provided fake_data."""
        
        idx = np.random.permutation(len(self.ori_data))
        train_idx = idx[:self.no]

        X_mb = [torch.tensor(self.ori_data[i][:-1, :(self.dim-1)], dtype=torch.float32).to(self.device) for i in train_idx]
        T_mb = [self.ori_time[i] - 1 for i in train_idx]
        Y_mb = [torch.tensor(np.reshape(self.ori_data[i][1:, (self.dim-1)], [len(self.ori_data[i][1:, (self.dim-1)]), 1]), dtype=torch.float32).to(self.device) for i in train_idx]
        
        X_mb = torch.nn.utils.rnn.pad_sequence(X_mb, batch_first=True, padding_value=0)
        Y_mb = torch.nn.utils.rnn.pad_sequence(Y_mb, batch_first=True, padding_value=0)

        fake_data = [torch.tensor(fake_data[i][:-1, :(self.dim-1)], dtype=torch.float32).to(self.device) for i in range(len(fake_data))]
        fake_t = [self.generated_time[i] - 1 for i in range(len(fake_data))]  
        fake_data = torch.nn.utils.rnn.pad_sequence(fake_data, batch_first=True, padding_value=0)

        with torch.no_grad():
            pred_fake_data = self.model(fake_data, torch.tensor(fake_t).cpu())  
        
        MAE_temp = 0
        for i in range(self.no):
            MAE_temp += torch.mean(torch.abs(Y_mb[i] - pred_fake_data[i]))  
            
        predictive_score = MAE_temp / self.no
        return predictive_score
