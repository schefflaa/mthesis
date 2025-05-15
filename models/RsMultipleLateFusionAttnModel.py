import torch
from torch import nn
import torch.nn.functional as F

DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

class RsMultipleLateFusionAttnModel(nn.Module):
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

        self.lstm = LateFusionAttnLSTM(
            in_dim  = channels*height*width,            # this the output size of the CNN per location
            strands = len(locations),                   # number of initially separate LSTM strands
            intermediate_dim = 128,                     # intermediate LSTM output size per location
            num_attn_blocks = 2,
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
        
class LateFusionAttnLSTM(nn.Module):
    def __init__(
            self, 
            in_dim=111_392,
            strands=1,
            intermediate_dim=128,
            num_attn_blocks=2,            
            out_dim=64,
            num_layers=1,
            cell_dropout=0.0,
            init=None,
            cam_embed_dim=16      # 👈 New: camera embedding dim
        ):

        super().__init__()
        self.num_layers = num_layers
        self.strands = strands
        self.hidden_dim = intermediate_dim
        self.hidden_dim_out = out_dim
        self.cam_embed_dim = cam_embed_dim
        self.init = init

        # Per-camera LSTMs
        for i in range(strands):
            setattr(self, f"lstm1_{i}", nn.LSTM(
                input_size  = in_dim,
                hidden_size = intermediate_dim,
                batch_first = True, 
                num_layers  = num_layers,
                dropout     = cell_dropout
            ))

        # Camera Embeddings
        self.cam_embeddings = nn.Embedding(strands, cam_embed_dim)

        # Attention Blocks
        self.attn_blocks = nn.Sequential(*[
            RsAttentionBlock(
                embed_dim=intermediate_dim + cam_embed_dim, 
                num_heads=4,
                ff_hidden_dim=(intermediate_dim + cam_embed_dim) * 2
            ) for _ in range(num_attn_blocks)
        ])

        # Learned Attention Pooling
        self.attn_pool_fc = nn.Linear(intermediate_dim + cam_embed_dim, 1)

        # Final LSTM
        self.lstm = nn.LSTM(
            input_size  = intermediate_dim + cam_embed_dim,
            hidden_size = out_dim, 
            batch_first = True, 
            num_layers  = num_layers,
            dropout     = cell_dropout
        )

    def forward(self, x):
        # Initialize hidden states
        h1 = [self.init_hidden(len(x), self.hidden_dim) for _ in range(self.strands)]
        h2 = self.init_hidden(len(x), self.hidden_dim_out)

        B, S, L, F = x.shape  # x: [B, seq, strands, feature_dim]

        # Per-camera LSTM
        lstm1_out = []
        for i in range(self.strands):
            lstm1 = getattr(self, f"lstm1_{i}")
            strand_in = x[:, :, i, :]    # [B, seq, feature_dim]
            strand_out, _ = lstm1(strand_in, h1[i])
            lstm1_out.append(strand_out)  # [B, seq, intermediate_dim]

        # Stack strands
        lstm1_out = torch.stack(lstm1_out, dim=2)  # [B, seq, strands, intermediate_dim]

        # Add Camera Embeddings
        cam_ids = torch.arange(self.strands, device=x.device)  # [0, 1, ..., strands-1]
        cam_embeds = self.cam_embeddings(cam_ids)  # [strands, cam_embed_dim]
        cam_embeds = cam_embeds.unsqueeze(0).unsqueeze(0).expand(B, S, -1, -1)  # [B, S, strands, cam_embed_dim]
        
        lstm1_out = torch.cat([lstm1_out, cam_embeds], dim=-1)  # [B, seq, strands, intermediate_dim + cam_embed_dim]

        # Attention blocks
        attn_in = lstm1_out.view(B * S, self.strands, -1)  # [B*S, strands, dim]
        attn_out = self.attn_blocks(attn_in)               # [B*S, strands, dim]

        # Attention-weighted pooling
        weights = self.attn_pool_fc(attn_out).squeeze(-1)  # [B*S, strands]
        weights = torch.softmax(weights, dim=1)            # softmax across strands
        pooled = (weights.unsqueeze(-1) * attn_out).sum(dim=1)  # [B*S, dim]

        pooled = pooled.view(B, S, -1)  # [B, seq, dim]

        # Final LSTM
        lstm2_out, _ = self.lstm(pooled, h2)  # [B, seq, out_dim]

        return lstm2_out

    def init_hidden(self, batch_size, hidden_dim):
        if not self.init or self.init == "zeros":
            z0 = torch.zeros(self.num_layers, batch_size, hidden_dim).to(DEVICE)
            c0 = torch.zeros(self.num_layers, batch_size, hidden_dim).to(DEVICE)
        elif self.init == "xavier_uniform":
            z0 = nn.init.xavier_uniform_(torch.empty(self.num_layers, batch_size, hidden_dim)).to(DEVICE)
            c0 = nn.init.xavier_uniform_(torch.empty(self.num_layers, batch_size, hidden_dim)).to(DEVICE)
        else:
            raise ValueError("Invalid initialization method.")
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
        
class RsAttentionBlock(nn.Module):
    def __init__(self, embed_dim=128, num_heads=4, ff_hidden_dim=256, dropout=0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads, batch_first=True)
        self.ln1 = nn.LayerNorm(embed_dim)
        self.dropout1 = nn.Dropout(dropout)

        self.ff = nn.Sequential(
            nn.Linear(embed_dim, ff_hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(ff_hidden_dim, embed_dim)
        )
        self.ln2 = nn.LayerNorm(embed_dim)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x):
        # Attention + residual + norm
        attn_out, _ = self.attn(x, x, x)
        x = self.ln1(x + self.dropout1(attn_out))

        # FFN + residual + norm
        ff_out = self.ff(x)
        x = self.ln2(x + self.dropout2(ff_out))
        return x
