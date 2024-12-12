# This runs the experiment for using 15 cmip models instead of 5 
# the model trained is GATLSTM with 256 embed dim (big) and 7 GAT layers
# rm -rf /scratch/users/yucli/enso_data/CMIP/multimesh1_processed
# python3 -m model_train --config model_configs/config_GATLSTM_multimesh_big_morecmip.py

# Lots of attension heads and smaller batches for current best model
rm -rf /scratch/users/yucli/enso_data/CMIP/multimesh1_processed
python3 -m model_train --config model_configs/config_GATLSTM_multimesh_big_3heads_smallbatch.py

# This runs the experiment for changing the embedding dimension 
# the model trained is GATLSTM with 5 GAT layers 
rm -rf /scratch/users/yucli/enso_data/CMIP/multimesh1_processed
python3 -m model_train --config model_configs/config_GATLSTM_multimesh_embdim32.py

rm -rf /scratch/users/yucli/enso_data/CMIP/multimesh1_processed
python3 -m model_train --config model_configs/config_GATLSTM_multimesh_embdim128.py
