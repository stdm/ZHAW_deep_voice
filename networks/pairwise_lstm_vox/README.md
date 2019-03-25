# Machine Learning for Speaker Clustering
Bachelor's Thesis - by Christian Lauener and Claude Lehmann

Named the pairwise network due to it's pairwise comparison nature

## Processing

#### Pre-Processing
- `common/spectrogram/main_data_extractor.py`: Extract mel-spectrograms from speaker audio files and save them to `data/training/VOXCELEB2_extracted` in .pickle file format

#### Processing (Training)
- `network_runner.py`: Wrapper functions to start training a model with appropriate parameters. In this case it's the network `src/nets/bilstm_2layer_dropout_plus_2dense.py` regarding the results from the bachelor's thesis ba17_stdm_1.

#### Post-Processing
- `keras_cluster_output_generator.py`: Execute clustering procedure on appropriate extracted speaker data with the previously trained network (-model)
- `clustering/cluster_tester.py`: Calculate missclassification rate and optionally show a scatter plot for processed clusterings

## Required folder structure
* data/experiments/logs/
* data/experiments/nets/
* data/experiments/plots/
* data/speaker_lists/
* data/training/VOXCELEB2/
* data/training/VOXCELEB2_extracted/