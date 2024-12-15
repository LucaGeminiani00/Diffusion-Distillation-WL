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

class DiscriminativeScoreModel():
    def __init__(self, ori_data, generated_data, iterations=2000, batch_size=128, lr=0.001, device=None):
        self.ori_data = ori_data
        self.generated_data = generated_data
        self.iterations = iterations
        self.batch_size = batch_size
        self.lr = lr
        self.device = device if device else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.is_trained = False 

        self.no, self.seq_len, self.dim = np.asarray(self.ori_data).shape
        self.ori_time, self.ori_max_seq_len = extract_time(self.ori_data)
        self.generated_time, self.generated_max_seq_len = extract_time(self.generated_data)
        
        self.hidden_dim = int(self.dim / 2)
        
        self.train_x, self.train_x_hat, self.test_x, self.test_x_hat, self.train_t, self.train_t_hat, self.test_t, self.test_t_hat = \
            train_test_divide(self.ori_data, self.generated_data, self.ori_time, self.generated_time)

        self.model = DiscriminatorModel(input_dim=self.dim, hidden_dim=self.hidden_dim).to(self.device)
        self.criterion = nn.BCEWithLogitsLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        
    def train(self):
        for itt in tqdm(range(self.iterations), desc='Training Discriminator', total=self.iterations):
            X_mb, T_mb = batch_generator(self.train_x, self.train_t, self.batch_size)
            X_hat_mb, T_hat_mb = batch_generator(self.train_x_hat, self.train_t_hat, self.batch_size)
            
            X_mb = torch.stack([torch.tensor(x, dtype=torch.float32).to(self.device) for x in X_mb])
            T_mb = torch.tensor(T_mb, dtype=torch.long).to(self.device)
            X_hat_mb = torch.stack([torch.tensor(x, dtype=torch.float32).to(self.device) for x in X_hat_mb])
            T_hat_mb = torch.tensor(T_hat_mb, dtype=torch.long).to(self.device)

            self.optimizer.zero_grad()

            y_logit_real = self.model(X_mb, T_mb.cpu())
            y_logit_fake = self.model(X_hat_mb, T_hat_mb.cpu())

            labels_real = torch.ones_like(y_logit_real)
            labels_fake = torch.zeros_like(y_logit_fake)

            d_loss_real = self.criterion(y_logit_real, labels_real)
            d_loss_fake = self.criterion(y_logit_fake, labels_fake)
            d_loss = d_loss_real + d_loss_fake

            d_loss.backward()
            self.optimizer.step()

        self.is_trained = True 
    
    def compute_dis(self, fake_data):
        if not self.is_trained: 
            raise("Discriminator Model must first be trained")
        test_x = torch.stack([torch.tensor(x, dtype=torch.float32).to(self.device) for x in self.test_x])
        test_t = torch.tensor(self.test_t, dtype=torch.long).to(self.device)
        
        fake_data = torch.stack([torch.tensor(x, dtype=torch.float32).to(self.device) for x in fake_data])
        fake_t = torch.tensor(self.test_t_hat, dtype=torch.long).to(self.device)  

        with torch.no_grad():
            y_pred_real_curr = torch.sigmoid(self.model(test_x, test_t.cpu()))
            y_pred_fake_curr = torch.sigmoid(self.model(fake_data, fake_t.cpu()))

        y_pred_real_curr_bin = (y_pred_real_curr > 0.5).int()
        y_pred_fake_curr_bin = (y_pred_fake_curr > 0.5).int()

        y_pred_real_curr_bin = y_pred_real_curr_bin.view(-1)
        y_pred_fake_curr_bin = y_pred_fake_curr_bin.view(-1)

        y_pred_final = torch.cat((y_pred_real_curr_bin, y_pred_fake_curr_bin), dim=0)
        y_label_final = torch.cat((torch.ones(y_pred_real_curr_bin.size(0)), torch.zeros(y_pred_fake_curr_bin.size(0))), dim=0)

        acc = accuracy_score(y_label_final.cpu(), y_pred_final.cpu())

        discriminative_score = np.abs(0.5 - acc)
        return discriminative_score