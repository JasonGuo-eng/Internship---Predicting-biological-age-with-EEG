#RNN
import torch
import torch.nn as nn
import torch.nn.functional as F

class EEGRNN(nn.Module):
    def __init__(self, input_size=7, hidden_size=64, num_layers=2, dropout=0.3):
        super(EEGRNN, self).__init__()

        # RNN expects input of shape [batch_size, seq_len, input_size]
        self.rnn = nn.LSTM(input_size=input_size,
                           hidden_size=hidden_size,
                           num_layers=num_layers,
                           batch_first=True,
                           dropout=dropout,
                           bidirectional=False)

        self.fc1 = nn.Linear(hidden_size, 32)
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(32, 1)  # Output: predicted age

    def forward(self, x):
        # x shape: [B, C, T] => [B, T, C]
        x = x.permute(0, 2, 1)

        # RNN output
        out, (hn, cn) = self.rnn(x)  # out: [B, T, hidden], hn: [num_layers, B, hidden]
        last_hidden = hn[-1]         # Take last layer's hidden state: [B, hidden]

        x = F.relu(self.fc1(last_hidden))
        x = self.dropout(x)
        x = self.fc2(x)

        return x

EEG_net = EEGRNN()

train_net(EEG_net, train_loader, val_loader, y_mean, y_std, batch_size=32, learning_rate=5e-3, num_epochs=70, checkpoint_freq=5)
model_path1 = get_model_name("EEG_net", batch_size=64, learning_rate=1e-3, epoch=70)
test_acc = model_test_accuracy(EEG_net, test_loader)
print(f"Test Accuracy: {test_acc * 100:.2f}%")
