# Pretraining parameters
EPOCHS = 4
LEARNING_RATE = 1e-4
EXPERIMENT_NAME = 'GCNRNN_simplegrid_small'
BATCH_SIZE = 16 # number of batches of 36

# Finetuning parameters
FT_EPOCHS = 5
FT_LEARNING_RATE = 1e-5
FT_BATCH_SIZE = 16
FT_EXPERIMENT_NAME = '5epochs_1e-5'

# Model parameters
GENET_EMB_DIM = 64
GENET_NUM_LAYERS = 5
GENET_DROPOUT = 0.3
ENC_HIDDEN_DIM = 64

# Model architecture
NODE_EMBEDDER = 'GCN' # GAT or GCN
ENC_DEC = 'RNN' # RNN or LSTM
ADJACENCY = 'simple_grid_dense' # Options: simple_grid, simple_grid_dense, multimesh1