| group | argument | definition |
|-------|----------|------------|
|dataset_args|dataset_name|dataset name, choose from 'nyc', 'tky', and 'ca'|
| |min_poi_freq|the least value of one poi's checkin records, if less than or equal to this value, we will remove this poi|
| |min_user_freq|the least value of one user's checkin records, if less than or equal to this value, we will remove this user|
| |session_time_interval| the time interval of consecutive checkin records in every trajectory should be larger than or equal to this value |
| |threshold| the similarity threshold of two trajectories when building hypergraph, if less than this value, we will remove this traj2traj relation |
| |filter_mode| the similarity metric, choose from 'jaccard' and 'min size'|
| |num_spatial_slots|the total number of slots for continuous distance value|
| |spatial_slot_type|construct distance slots automatically based on min, max value of distance, choose from 'linear' and 'exp'|
| |do_label_encode| whether to encode the id via LabelEncoder|
| |only_last_metric| whether to use only the last checkin of every trajectory as sample to evaluate our model|
| |max_d_epsilon| add this value to maximum distance to avoid bugs|
|model_args|model_name|model name, choose from 'sthgcn' (our model) and 'seq_transformer' (for ablation study)|
| |intra_jaccard_threshold|the intra-user similarity threshold for hyperedge2hyperedge collaboration, only keep those collaborations whose similarities are larger than this value|
| |inter_jaccard_threshold|the inter-user similarity threshold for hyperedge2hyperedge collaboration, only keep those collaborations whose similarities are larger than this value
| |sizes|sample size for different hops, the last element is for checkin2trajectory, other elements is for multi-hop trajectory2trajectory. e.g. sizes=[10, 20, 30], [10,20] is for traj2traj 2-hop sampling, [30] is for ci2traj.|
| |dropout_rate|the dropout rate|
| |num_edge_type|the total number of edge type|
| |generate_edge_attr|whether to generate edge attr embedding based on edge type|
| |embed_fusion_type|embedding fusion type, choose from 'concat' and 'add'|
| |embed_size|the embedding size of id embedding and the hidden representation of trajectory|
| |st_embed_size|the embedding size of spatial and temporal embedding|
| |activation|the activation function, choose from 'elu', 'relu', 'leaky_relu' and 'tanh'|
| |phase_factor|phase factor for time encoder|
| |use_linear_trans|whether to use linear transformation before output for time encoder|
| |do_traj2traj|whether to use hyperedge2hyperedge collaboration|
| |distance_encoder_type|encoder type of distance, choose from 'time', 'hstlstm', 'stan' and 'simple'. Specially, 'time' means using the TimeEncoder to handle distance value|
| |quantile|clip the maximum distance value with clip(0, max_d*quantile), should modify the code in dataset/lbsn_dataset to make this work|
|conv_args|num_attention_heads|the total number of attention heads|
| |residual_beta|the residual weight of initial representation for gated residual module|
| |learn_beta|whether to learn residual beta automatically|
| |conv_dropout_rate|the dropout rate for hypergraph transformer|
| |trans_method|the translation method of message assembler, choose from 'corr', 'sub', 'add', 'multi' and 'concat'|
| |edge_fusion_mode|the fusion mode of edge vector, choose from 'concat' and 'add'|
| |head_fusion_mode|the fusion mode of multi-head, choose from 'concat' and 'add'|
| |time_fusion_mode|the fusion mode of time vector, choose from 'concat' and 'add'|
| |residual_fusion_mode|the fusion mode of gated residual module, choose from 'concat' and 'add'|
| |negative_slope|the negative slope for leaky_relu activation function|
|run_args|seed|random seed for generate random number, and reproduce the experiments. Not used for multiple-run setting.|
| |gpu|gpu index, use cpu if set -1|
| |batch_size|training batch size|
| |eval_batch_size|evaluation batch size|
| |learning_rate|the learning rate|
| |do_train|whether to do training|
| |do_validate|whether to do validation|
| |do_test|whether to do testing|
| |warm_up_steps|the warm up steps with constant initial learning rate|
| |cooldown_rate|the cooldown rate for learning rate schedualing, make the learning rate approximate an exponential decay curve with respect to the global steps|
| |max_steps|the max steps for training|
| |epoch|the training epoch|
| |valid_steps|do evaluating every valid_steps|
| |num_workers|the total number of workers for dataloader|
| |init_checkpoint|the checkpoint path|
|seq_transformer_args|only works when `model_args.name==seq_transformer`| |
| |sequence_length |the max length of the sequences|
| |header_num|the head number of multi-head|
| |encoder_layers_num|the total number of encoder layers|
| |hidden_size|the embedding size of hidden representation|
| |dropout|the dropout rate|
| |do_positional_encoding|whether to use positional encoding|
