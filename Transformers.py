#transformer
import torch
import torch.nn as nn

class EEGTransformer(nn.Module):
    def __init__(self, input_channels=7, seq_len=179, d_model=64, nhead=4, num_layers=2, dropout=0.3):
        super(EEGTransformer, self).__init__()

        self.input_proj = nn.Linear(input_channels, d_model)  # project channels to d_model
        self.pos_embedding = nn.Parameter(torch.randn(1, seq_len, d_model))

        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dropout=dropout)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(seq_len * d_model, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 1)  # Output a single age prediction
        )

    def forward(self, x):
        # x: [B, C, T] → [B, T, C]
        x = x.permute(0, 2, 1)

        x = self.input_proj(x)  # [B, T, d_model]
        x = x + self.pos_embedding  # Add positional encoding

        x = self.transformer_encoder(x)  # [B, T, d_model]

        out = self.fc(x)  # Final regressor
        return out.squeeze()

EEG_net = EEGTransformer()
train_net(EEG_net, train_loader, val_loader, y_mean, y_std, batch_size=64, learning_rate=1e-3, num_epochs=70, checkpoint_freq=5)
model_path1 = get_model_name("EEG_net", batch_size=64, learning_rate=1e-3, epoch=70)
