# Machine Learning for Speaker Clustering
Bachelor's Thesis - by Lukic and Vogt

## Processing

#### Pre-Processing
- `common/spectrogram/main_data_extractor.py`: Extract mel-spectrograms from speaker audio files and save them to `data/training/TIMIT_extracted` in .pickle file format

#### Processing (Training)
- `clustering_network.py`: Run this file. It trains and saves the network to a place of your liking.
- `network_runner.py`: Wrapper functions to start training a model with appropriate parameters. In this case it's the network `src/nets/bilstm_2layer_dropout_plus_2dense.py` regarding the results from the bachelor's thesis ba17_stdm_1.

#### Post-Processing
- `clustering/cluster_tester.py`: Calculate missclassification rate and optionally show a scatter plot for processed clusterings
- `analyse/analyse_network.py`: Execute a trained network with train and test data, shows a scatter plot for the clustering output
