import numpy as np
import torch
import torch.nn as nn
import torch_geometric
import torch_geometric.nn
import lightning as L
from torch_geometric.nn import GCNConv
from torch_geometric.nn import global_add_pool, global_mean_pool

START = 10 # Start token outside of usual range of ONI scale
SEQ_LEN = 36

if torch.cuda.is_available():
    device = torch.device("cuda:0")  # Using the first CUDA device
    print('Using GPU')
else:
    device = torch.device("cpu")
    print('Using CPU')

# Node embedding network
# class GCN(torch.nn.Module):
class GCN(L.LightningModule):

    def __init__(self, input_dim=4, emb_dim=64, num_layers=3, dropout=0.3):

      super(GCN, self).__init__()

      self.convs = torch.nn.ModuleList([GCNConv(in_channels=input_dim, out_channels=emb_dim)] +
                  [GCNConv(in_channels=emb_dim, out_channels=emb_dim) for i in range(num_layers - 1)])
      self.bns = torch.nn.ModuleList([torch.nn.BatchNorm1d(num_features = emb_dim) for i in range(num_layers - 1)])
      self.softmax = torch.nn.LogSoftmax()
      self.dropout = dropout
      self.pool = global_mean_pool

    def reset_parameters(self):
      for conv in self.convs:
          conv.reset_parameters()
      for bn in self.bns:
          bn.reset_parameters()

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
The decoder takes in the final hidden state from the encoder (batch x h_e). It predicts ONI scores for a set number of 
months into the future. 
'''
# class dec(nn.Module):
class dec(L.LightningModule):
    def __init__(self, hidden_dim, output_len): # hidden dim is 2(h_e)
        super(dec, self).__init__()
        self.rnn_cell = nn.RNNCell(1, hidden_dim) # +1 to concatenate predictions
        self.linear_out = nn.Linear(hidden_dim, 1)
        self.output_len = output_len

    def forward(self, initial_input):
        outputs = []

        hidden = initial_input # input should be batch x (2 x h_e) (just hidden states)
        inp = torch.full((hidden.size(0), 1), START, dtype=torch.float32).to(self.device)
        
        for _ in range(self.output_len):
            hidden = self.rnn_cell(inp, hidden)
            output = self.linear_out(hidden)
            outputs.append(output)
            inp = output
        
        final = torch.stack(outputs, dim=1)  # # batch, seq_len, 1
        final = final.permute(0, 2, 1)  # change shape to batch, 1, seq_len. ensures that final oni values are in sets of 36
        final = final.reshape(-1)  # Now reshape to batch * seq_len
        return final


'''
Given a batch x seq_len x node_count x node_dim matrix, return a batch x seq_len matrix containing ONI predictions.

For this baseline model, we have a defined input len (seq_len = 32) and output length (output_length=32). 

GNNRNN first uses a GCN to create node embeddings for each of the 1728 (lat x long, 24 x 72) nodes in the world graph. 
Then, each graph's set of node embeddings are passed through a linear layer to produce a graph embedding, yielding a 
batch x seq_len x graph_emb_dim matrix. This is finally fed into the encoder-decoder RNN architecture to produce predictions.

The entire model is end-to-end differentiable.
'''
# class GNNRNN(nn.Module):
class GNNRNN(L.LightningModule):
    # graph_emb_dim = graph embedding dimension; hidden_dim: = encoder hidden dim
    def __init__(self, graph_emb_dim, hidden_dim, output_length, device):

        super(GNNRNN, self).__init__()
        self.ge = GCN(emb_dim=graph_emb_dim).to(device)
        self.encoder = nn.RNN(input_size=graph_emb_dim, hidden_size=hidden_dim, batch_first=True, bidirectional=True).to(device)
        self.decoder = dec(2 * hidden_dim, SEQ_LEN).to(device)
        self.output_length = output_length

    # x will be (batch x seq_len) x emb_dim 
    def forward(self, x):
        graph_embds = self.ge(x) # (batch x seq_len) x dim
        batched_graph_embds = graph_embds.reshape(graph_embds.size(0) // 36, 36, graph_embds.size(-1)) # batch x seq_len x feature_dim
        _, temp = self.encoder(batched_graph_embds) # temp: 2 x batch x emb_dim. 
        encoded = temp.transpose(0, 1).reshape(temp.size(1), -1).to(self.device) # encoded: batch x (emb_dim x 2)
        outputs = self.decoder(encoded)
        return outputs

    def training_step(self, batch, batch_idx):
        i, t = batch
        inputs = i.to(device)
        targets = t.to(device)
        outputs = self(inputs)
        criterion = torch.nn.MSELoss()
        loss = criterion(outputs, targets.type(torch.float32)).item() # no idea why the conversion in the cmip class isn't working.
        self.log("train_loss", loss)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-5)