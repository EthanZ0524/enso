We have implemented several custom PyG DataSet and DataLoader classes to handle data processing/loading. Details and design choices are expanded upon below. 

**Data**: 
We use CMIP and SODA datasets to train and fine-tine our models, respectively. They are described in https://spj.science.org/doi/full/10.34133/olar.0012 and can be downloaded at https://tianchi.aliyun.com/dataset/98942. 

Concretely, the CMIP and SODA datasets are NetCDF files consisting of a Nx36x24x72x4 array (N x 36 months x 24 latitude x 72 longitude x 4 variables). Each 'N' represents either a climate modeling simulation (CMIP) or a reanalysis sample. Each sample comprises of 36 monthly snapshots, each of which contains values for 4 variables (sst, t300, ua, va). 

The label dataset dimension is (N x 36 months). Each month's label corresponds to the ONI value for the *following* month. (For example: at month=1 (January), the label value corresponds to the average SST anomaly in the Nino3.4 region (5N-5S, 120W-170W) across January, February, and March, eg. the ONI value for February). 

**PyG DataSet Implementation**:
We chose to implement our PyG DataSet classes such that they would draw individual timesteps from our datasets, where a single timestep is defined as a 1728-node homogeneous undirected graph with nodes containing a 4-dimensional feature vector. 

Concretely, drawing a sample with index i using our DataSet classes results in the timestep corresponding to N = (i // 36) and month = (i % 36). In each node, features are ordered as \[sst, t300, ua, va\]. Nodes themselves are ordered such that the nth node's latitude index is (n // 72) and longitude index is (n % 72). The DataSet saves each of these individual graphs as a .pt file in the 'processed' folder.

Labels are saved as class attributes of the DataSet classes, and are accessed within the Lightning Model.

**PyG DataLoader Implementation**:
Our DataSet implementation (drawing individual graphs as opposed to sets of 36) was done with the DataLoader's minibatching feature in mind. However, as a result, we cannot shuffle data samples randomly. Instead, they must be shuffled in sets of 36 to maintain temporal ordering. To satisfy both constraints, we use a Torch Sampler class - this still allows the PyG DataLoader to minibatch each 36-graph set into one single large graph, while also being able to shuffle sets of 36 and maintain their internal ordering. 