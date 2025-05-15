import torch
from torch import nn
import torch.nn.functional as F

DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

class RsMultipleLateCombinationModel(nn.Module):
    def __init__(self, img_size=(3, 250, 250), locations=None, lstm_out_size=64, num_conv_blocks=2, final_actfn=torch.nn.Identity(), conv_init=None, lstm_init=None, fc_init=None):
        super().__init__()

        self.cnn = RsCNN(
            img_size            = img_size,
            num_conv_blocks     = num_conv_blocks,
            conv_kernel_size    = 3,
            activation_fn       = F.relu,
            init                = conv_init
        )
        
        # Compute the output size of the CNN
        height = (img_size[1] - 4 * (2**num_conv_blocks - 1)) // (2**num_conv_blocks)
        width  = (img_size[2] - 4 * (2**num_conv_blocks - 1)) // (2**num_conv_blocks)
        channels = 2**(num_conv_blocks + 3)

        self.lstm = RsLateCombLSTM(
            in_dim  = channels*height*width,            # this the output size of the CNN per location
            strands = len(locations),                   # number of initially separate LSTM strands
            intermediate_dim = int(128//len(locations)),    # intermediate LSTM output size per location
            out_dim = lstm_out_size,
            init    = lstm_init
        )

        self.fc = nn.Linear(lstm_out_size, 1)
        self.activation_fn = final_actfn
        self.init_weights(fc_init)

    def forward(self, x_img, _):
        # handling image sequences in cnn: https://discuss.pytorch.org/t/how-to-input-image-sequences-to-a-cnn-lstm/89149/2
        
        # x_img has shape: [B, seq, loc, C, H, W]
        bat_len = x_img.shape[0]
        seq_len = x_img.shape[1]
        loc_len = x_img.shape[2]
        
        # CNN
        cnn_in   = x_img.view(-1, *x_img.shape[3:])             # reshape input to [B*seq*L, C, H, W] for the CNN
        cnn_out  = self.cnn(cnn_in)              

        # LSTM
        lstm_in = cnn_out.view(bat_len, seq_len, loc_len, -1)    # reshape the CNN output to [B, seq, loc, L*C*H*W] for the LSTM
        lstm_out = self.lstm(lstm_in)                          

        # FC
        fc_in    = lstm_out[:, -1, :]                           # reshape the LSTM output to [B, LSTM_out_dim] for the FC layer
        out      = self.fc(fc_in)                               # ↪ this uses only the last time step's output 
        return self.activation_fn(out)                          # ↪ (i.e. the final contextualized representation of the sequence)
    
    def init_weights(self, fc_init):
        """Initialize weights of the fully connected layer."""

        if fc_init is None:
            return  # defaults to Xavier Uniform Initialization
        elif fc_init == "kaiming_normal": 
            nn.init.kaiming_normal_(self.fc.weight)            
        elif fc_init == "kaiming_uniform": 
            nn.init.kaiming_uniform_(self.fc.weight)
        elif fc_init == "xavier_normal": 
            nn.init.xavier_normal_(self.fc.weight)
        elif fc_init == "xavier_uniform":
            nn.init.xavier_uniform_(self.fc.weight)
        else:
            raise ValueError("Invalid initialization method. Use 'kaiming/xavier' + '_uniform/_normal'.")
        
class RsLateCombLSTM(nn.Module):
    def __init__(
            self, 
            in_dim=111_392,
            strands=1,
            intermediate_dim=128,
            out_dim=64,
            num_layers=1,
            cell_dropout=0.0,
            init=None
        ):

        super().__init__()
        self.num_layers = num_layers
        self.strands = strands	
        self.hidden_dim = intermediate_dim
        self.hidden_dim_out = out_dim
        self.init = init

        for i in range(strands):
            setattr(self, f"lstm1_{i}", nn.LSTM(
                input_size  = in_dim,
                hidden_size = intermediate_dim,
                batch_first = True, 
                num_layers  = num_layers,
                dropout     = cell_dropout
            ))

        self.lstm = nn.LSTM(
            input_size  = intermediate_dim * self.strands,
            hidden_size = out_dim, 
            batch_first = True, 
            num_layers  = num_layers,
            dropout     = cell_dropout
        )

    def forward(self, x):
        # Initialize hidden states
        h1 = [self.init_hidden(len(x), self.hidden_dim) for _ in range(self.strands)]
        h2 = self.init_hidden(len(x), self.hidden_dim_out)

        # x has shape [B, seq, loc, L*C*H*W]
        lstm1_out = []
        for i in range(self.strands):
            lstm1 = getattr(self, f"lstm1_{i}")                     # get the LSTM for the strand
            lstm1_in = x[:, :, i, :]                                # get the input for the strand, shape [B, seq, C*H*W]
            lstm1_out_i, (zt1, ct1) = lstm1(lstm1_in, h1[i])        # run the LSTM strand
            lstm1_out.append(lstm1_out_i)                           # append the output of the strand

        # lstm1_out is a list (length == self.strands) of tensors with shape [B, seq, intermediate_dim]
        lstm1_out = torch.cat(lstm1_out, dim=2)                     # concatenate the outputs of the strands along the feature dimension
        # lstm1_out has shape [B, seq, intermediate_dim * self.strands]
        lstm2_out, (zt2, ct2) = self.lstm(lstm1_out, h2)
        return lstm2_out

    def init_hidden(self, batch_size, hidden_dim):
        """Initialize hidden states.

        Returns a tuple of two num_layers x batch_size x hidden_dim tensors (one for
        initial cell states, one for initial hidden states) consisting of all zeros.
        """
        if not self.init or self.init == "zeros":
            z0 = torch.zeros(self.num_layers, batch_size, hidden_dim).to(DEVICE)
            c0 = torch.zeros(self.num_layers, batch_size, hidden_dim).to(DEVICE)
        elif self.init == "xavier_uniform":
            z0 = nn.init.xavier_uniform_(torch.empty(self.num_layers, batch_size, hidden_dim)).to(DEVICE)
            c0 = nn.init.xavier_uniform_(torch.empty(self.num_layers, batch_size, hidden_dim)).to(DEVICE)
        else:
            raise ValueError("Invalid initialization method. Use 'xavier_uniform' or 'zeros'.")

        return (z0, c0)

class RsCNN(nn.Module):
    def __init__(
        self,
        img_size=(3, 250, 250),
        num_conv_blocks=2,
        conv_kernel_size=3,
        activation_fn=F.relu,
        init=None
    ):
        super().__init__()
        self.conv_blocks = nn.ModuleList()

        for i in range(num_conv_blocks):
            self.conv_blocks.append(
                RsConvBlock(
                    in_channels      = img_size[0] if i == 0 else 2**(i+3), #16*i,
                    out_channels     = 2**(i+4),
                    conv_kernel_size = conv_kernel_size, 
                    pool_kernel_size = 2,
                    dropout_rate     = 0.3 if i == 0 else 0.1,
                    activation_fn    = activation_fn,
                    init             = init
                )
            )
        
    def forward(self, x):
        for block in self.conv_blocks:
            x = block(x)
        return x

class RsConvBlock(nn.Module):
    def __init__(
        self,
        in_channels=3,
        out_channels=16,
        conv_kernel_size=3,
        pool_kernel_size=2,
        dropout_rate=0.3,
        activation_fn=F.relu,
        init=None
    ):
        super().__init__()
        self.init = init

        self.conv1 = nn.Conv2d(
            in_channels  = in_channels,
            out_channels = out_channels,
            kernel_size  = conv_kernel_size
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(
            in_channels  = out_channels,
            out_channels = out_channels,
            kernel_size  = conv_kernel_size
        )
        self.pool = nn.MaxPool2d(pool_kernel_size)
        self.dropout = nn.Dropout(dropout_rate)
        self.activation_fn = activation_fn
        self.init_weights()

    def forward(self, x):
        out_c1 = self.activation_fn(self.conv1(x))
        out_bn = self.bn(out_c1)
        out_c2 = self.activation_fn(self.conv2(out_bn))
        out_pool = self.pool(out_c2)
        out = self.dropout(out_pool)
        return out

    def init_weights(self):
        """Initialize weights of the convolutional layers."""
        if self.init is None:
            return  # defaults to Kaiming Uniform Initialization
        elif self.init == "kaiming_normal": 
            nn.init.kaiming_normal_(self.conv1.weight)
            nn.init.kaiming_normal_(self.conv2.weight)      
        elif self.init == "kaiming_uniform": 
            nn.init.kaiming_uniform_(self.conv1.weight)
            nn.init.kaiming_uniform_(self.conv2.weight)
        elif self.init == "xavier_normal": 
            nn.init.xavier_normal_(self.conv1.weight)
            nn.init.xavier_normal_(self.conv2.weight)
        elif self.init == "xavier_uniform":
            nn.init.xavier_uniform_(self.conv1.weight)
            nn.init.xavier_uniform_(self.conv2.weight)
        else:
            raise ValueError("Invalid initialization method. Use 'kaiming/xavier' + '_uniform/_normal'.")
        