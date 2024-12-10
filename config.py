# Pretraining parameters
EPOCHS = 10
LEARNING_RATE = 1e-4
EXPERIMENT_NAME = 'GATLSTM_newdata_large'
BATCH_SIZE = 16 # number of batches of 36

# Finetuning parameters
FT_EPOCHS = 5
FT_LEARNING_RATE = 1e-5
FT_BATCH_SIZE = 16
FT_EXPERIMENT_NAME = '5epochs_1e-5'

# Model parameters
GENET_EMB_DIM = 128
GENET_NUM_LAYERS = 7
GENET_DROPOUT = 0.3
ENC_HIDDEN_DIM = 128

# Model architecture
NODE_EMBEDDER = 'GAT' # GAT or GCN
ENC_DEC = 'LSTM'