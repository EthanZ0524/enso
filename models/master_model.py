import numpy as np
import torch
import torch.nn as nn
import torch_geometric
import torch_geometric.nn
from torch_geometric.nn import GATConv, GCNConv
import pytorch_lightning as pl
from torch_geometric.nn import global_add_pool, global_mean_pool

START = 10 # Start token outside of usual range of ONI scale
NUM_VARS = 4

'''
Graph embedding network supporting GCNConv and GATConv layers. 

Params:
    hidden_dim (int): dimension of final graph embedding
    num_layers (int): number of GCN/GAT layers
    dropout (float):  dropout rate for GCN/GAT layers
    node_embedder (str): what kind of layer to use for message passing. Options:
        'GCN'
        'GAT'
    node_dims: dimension of each node in input graphs

Shapes: 
    forward input: a PyG Batch object of (36 * batch) graphs.
    forward output: batch * 36 * hidden_dim
'''
class GENet(pl.LightningModule):
    def __init__(self,  
        hidden_dim, 
        num_layers, 
        dropout,
        node_embedder, 
        node_dim=NUM_VARS,
    ):

        super(GENet, self).__init__()
        if node_embedder == 'GCN':
            self.node_embedder = GCNConv
        elif node_embedder == 'GAT':
            self.node_embedder = GATConv
        else:
            ValueError('Node embedder type not recognized.')

        self.convs = torch.nn.ModuleList([self.node_embedder(in_channels=node_dim, out_channels=hidden_dim)] +
            [self.node_embedder(in_channels=hidden_dim, out_channels=hidden_dim) for i in range(num_layers - 1)])

        self.bns = torch.nn.ModuleList([torch.nn.BatchNorm1d(num_features = hidden_dim) for i in range(num_layers - 1)])
        self.dropout = dropout
        self.pool = global_mean_pool

    def forward(self, batched_data):
      x, edge_index, batch = batched_data.x.type(torch.float32), batched_data.edge_index, batched_data.batch
      for i in range(len(self.bns)):
        x = self.convs[i].forward(x, edge_index)
        x = self.bns[i](x)
        x = torch.nn.functional.relu(x)
        x = torch.nn.functional.dropout(x)
      out = self.convs[-1].forward(x, edge_index)
      out = self.pool(out, batch)
      return out

'''
RNN Decoder to produce final ONI predictions.

Params:
    hidden_dim (int): dimension of RNN hidden state
    output_len (int): number of ONI values to forecast

Shapes:
    forward input: (2, batch, h_e) (h_e is the hidden state dimension of the previous RNN encoder)
    forward output: 
'''
class RNNDec(pl.LightningModule):
    def __init__(self, hidden_dim, output_len): # hidden dim is 2(h_e)
        super(RNNDec, self).__init__()
        self.cell = nn.RNNCell(1, hidden_dim) 
        self.linear_out = nn.Linear(hidden_dim, 1)
        self.output_len = output_len

    def forward(self, initial_input):
        outputs = []
        hidden = initial_input.transpose(0, 1).reshape(temp.size(1), -1).to(self.device) # batch x (emb_dim x 2)
        inp = torch.full((hidden.size(0), 1), START, dtype=torch.float32).to(self.device)
        
        for _ in range(self.output_len):
            hidden = self.cell(inp, hidden)
            output = self.linear_out(hidden)
            outputs.append(output)
            inp = output
        
        final = torch.stack(outputs, dim=1)  # # batch, seq_len, 1
        final = final.squeeze(-1)  # batch_size, seq_len
        return final

'''
LSTM Decoder to produce final ONI predictions.

Params:
    hidden_dim (int): dimension of LSTM hidden state
    output_len (int): number of ONI values to forecast

Shapes:
'''
class LSTMDec(pl.LightningModule):
    def __init__(self, hidden_dim, output_len): # hidden_dim is 2(h_e)
        super(LSTMDec, self).__init__()
        self.cell = nn.LSTMCell(1, hidden_dim)  # +1 to concatenate predictions if needed
        self.linear_out = nn.Linear(hidden_dim, 1)
        self.output_len = output_len

    def forward(self, initial_hidden):
        """
        Args:
            initial_hidden: tuple (h_0, c_0)
                - h_0: Initial hidden state (batch_size x hidden_dim)
                - c_0: Initial cell state (batch_size x hidden_dim)
        """
        outputs = []

        # Unpack the initial hidden and cell states
        h_n, c_n = initial_hidden # both (2, batch, h_e)
        h_t = h_n.permute(1, 0, 2).reshape(h_n.size(1), -1)  # (batch_size, 2 * emb_dim)
        c_t = c_n.permute(1, 0, 2).reshape(c_n.size(1), -1)  # (batch_size, 2 * emb_dim)
        
        # Initialize input to start token
        inp = torch.full((h_t.size(0), 1), START, dtype=torch.float32, device=self.device)

        for _ in range(self.output_len):
            # Update hidden and cell states
            h_t, c_t = self.cell(inp, (h_t, c_t))
            
            # Compute output
            output = self.linear_out(h_t)  # Shape: (batch_size, 1)
            outputs.append(output)
            
            # Feedback loop: set input for the next timestep to the current output
            inp = output

        # Stack outputs into a single tensor
        final = torch.stack(outputs, dim=1)  # Shape: (batch_size, seq_len, 1)
        final = final.squeeze(-1)  # Shape: (batch_size, seq_len)
        return final

'''
Given a batch x seq_len x node_count x node_dim matrix, return a batch x seq_len matrix containing ONI predictions.

For this baseline model, we have a defined input len (seq_len = 32) and output length (output_length=32). 

GNNRNN first uses a GCN to create node embeddings for each of the 1728 (lat x long, 24 x 72) nodes in the world graph. 
Then, each graph's set of node embeddings are passed through a linear layer to produce a graph embedding, yielding a 
batch x seq_len x graph_emb_dim matrix. This is finally fed into the encoder-decoder RNN architecture to produce predictions.

The entire model is end-to-end differentiable.
'''
class MasterModel(pl.LightningModule):
    # graph_emb_dim = graph embedding dimension; hidden_dim: = encoder hidden dim
    def __init__(self, 
        graph_emb_dim, 
        ge_net_layers, 
        ge_net_dropout, 
        node_embedder: str, # GAT, GCN, etc.
        enc_hidden_dim, 
        enc_dropout,
        enc_dec: str, # LSTM, RNN, etc.
        output_length, 
        lr
    ):
    
        super(MasterModel, self).__init__()
        self.save_hyperparameters()
        self.shuffle_indices = None # used if we need to shuffle data and labels
        self.learning_rate = lr
        self.labels = None

        self.ge = GENet(hidden_dim=graph_emb_dim, 
            num_layers=ge_net_layers, 
            dropout=ge_net_dropout,
            node_embedder=node_embedder)

        encoder_type = decoder_type = None

        if enc_dec == 'RNN':
            encoder_type = nn.RNN
            decoder_type = RNNDec
        elif enc_dec == 'LSTM':
            encoder_type = nn.LSTM
            decoder_type = LSTMDec
        else:
            ValueError('Encoder decoder type not recognized.')

        self.encoder = encoder_type(input_size=graph_emb_dim, 
            hidden_size=enc_hidden_dim,
            dropout=enc_dropout, 
            batch_first=True, 
            bidirectional=True)

        self.decoder = decoder_type(hidden_dim=2*enc_hidden_dim, 
            output_len=output_length)

    def forward(self, x):
        graph_embds = self.ge(x)
        batched_graph_embds = graph_embds.reshape(graph_embds.size(0) // 36, 36, graph_embds.size(-1)) # batch x seq_len x feature_dim
        _, encoded = self.encoder(batched_graph_embds) # encoded: 2 x batch x emb_dim
        outputs = self.decoder(encoded)
        return outputs

    def training_step(self, batch, batch_idx):
        n = self.trainer.datamodule.sampler.group_size // 36 # how many groups of 36 we got per batch
        if batch_idx == 0: # triggered @ start of every epoch
            if hasattr(self.trainer.datamodule.sampler, 'shuffle_indices'):
                self.shuffle_indices = self.trainer.datamodule.sampler.shuffle_indices

            labels = self.trainer.datamodule.dataset.get_labels() # x * 24
            if self.shuffle_indices is not None:
                labels = labels[self.shuffle_indices]
            self.labels = labels[:self.trainer.datamodule.sampler.num_batches * n] # throwing out unused labels; rounded(x) * 24

        batch_labels = torch.from_numpy(self.labels[batch_idx*n:batch_idx*n + n])

        inputs = batch.to(self.device)
        targets = batch_labels.to(self.device)

        outputs = self(inputs)
        criterion = torch.nn.MSELoss()
        loss = criterion(outputs, targets.type(torch.float32))
        self.log("train_loss", loss.item(), on_step=True, on_epoch=True, prog_bar=True, batch_size=self.trainer.datamodule.sampler.group_size)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate)