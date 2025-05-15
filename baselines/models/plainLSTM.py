import torch
from torch import nn
import torch.nn.functional as F

DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

class plainLSTMFC(nn.Module):
    def __init__(
            self, 
            locations,
            out_dim=64,
            final_actfn=torch.nn.Identity()
        ):

        super().__init__()
        
        self.hidden_dim = out_dim
        self.num_layers = 1

        self.lstm = nn.LSTM(
            input_size  = 1, # 1 is for the pv_output_history (1 feature per time step)
            hidden_size = out_dim, 
            batch_first = True, 
            num_layers  = self.num_layers,
            dropout     = 0.0,
        )
        self.fc = nn.Linear(out_dim, 1)
        self.activation_fn = final_actfn

    def forward(self, pv_output_history):
        h                       = self.init_hidden(len(pv_output_history), self.hidden_dim)
        lstm_in                 = pv_output_history.unsqueeze(-1)               # add a dummy dimension to the end of the tensor
        lstm_out, (zt2, ct2)    = self.lstm(lstm_in, h)                         # lstm_in.shape = [batch, seq, 1]
        fc_in                   = lstm_out[:, -1, :]                            # reshape the LSTM output to [B, LSTM_out_dim] for the FC layer
        out                     = self.fc(fc_in)                                # ↪ this uses only the last time step's output 
        return self.activation_fn(out)                                          # ↪ (i.e. the final contextualized representation of the sequence)

    def init_hidden(self, batch_size, hidden_dim):
        z0 = torch.zeros(self.num_layers, batch_size, hidden_dim).to(DEVICE)
        c0 = torch.zeros(self.num_layers, batch_size, hidden_dim).to(DEVICE)
        return (z0, c0)