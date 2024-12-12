# Pretraining parameters
EPOCHS = 4
LEARNING_RATE = 1e-4
EXPERIMENT_NAME = 'GATLSTM_multimesh_embdim32'
BATCH_SIZE = 16 # number of batches of 36
NUM_MODELS = 5
NUM_HEADS = 1

# Finetuning parameters
FT_EPOCHS = 5
FT_LEARNING_RATE = 1e-5
FT_BATCH_SIZE = 16
FT_EXPERIMENT_NAME = '5epochs_1e-5'

# Model parameters
GENET_EMB_DIM = 32
GENET_NUM_LAYERS = 5
GENET_DROPOUT = 0.3
ENC_HIDDEN_DIM = 32

# Model architecture
NODE_EMBEDDER = 'GAT' # GAT or GCN
ENC_DEC = 'LSTM' # RNN or LSTM
ADJACENCY = 'multimesh1' # Options: simple_grid, simple_grid_dense, multimesh1