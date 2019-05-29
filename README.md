 # ZHAW deep voice
 
 The ZHAW deep voice is a package of multiple neural networks that try resolving the speaker clustering task. The goal is to provide a uniform way of data-access, -preprocession and analysis fo the results.
 
## Data sets
The suite supports two datasets at this point. The TIMIT dataset and the VoxCeleb2 dataset.
Further datasets may be added. To do so, you need to edit the common/extrapolation/speaker.py and common/extrapolation/speaker_factory.py to specify handling of dataset specifics.
The speaker list files are located in common/data/speaker_lists.

### TIMIT
This is a paid product from the LDC and can be [obtained here.](https://www.ldc.upenn.edu/).
This data also needs to be processed using the [sph2pipe tool](https://www.ldc.upenn.edu/language-resources/tools/sphere-conversion-tools) and be put in the folder common/data/training/TIMIT.

### VoxCeleb2
This is a freely available dataset set and can be [obtained here.](http://www.robots.ox.ac.uk/~vgg/data/voxceleb/vox2.html). It contains m4a-format audiofiles which need to be converted to .wav files to enable processing. You may use the audio_converter.py script to do so or alternatively use a tool of your choice (e.g. [FFmpeg](https://ffmpeg.org/)).
The converted data needs to be put in the folder common/data/training/VOXCELEB2.

 ## Using deep voice
 If you simply want to use it, you can let docker do the work for you and let it import all needed packages.
 
 In any way, whether you fork and pull the source code or let docker handle it for you, the whole suite is controllable over a one file interface, controller.py.

 To view all available parametrs call

 ```
 controller.py -help
 ```

Further arguments are: 
```
controller.py
  -h, --help         show this help message and exit
  -setup             Run project setup.
  -n NETWORK         The network to use for training or analysis. Available: 'pairwise_lstm', 'pairwise_lstm_vox', 'arc_face', 'pairwise_kldiv', 'flow_me' and 'luvo'. All networks use different sets of parameters, so make sure to check their individual implementation
  -train             Train the specified network.
  -test              Test the specified network.
  -clear             Clean directories before starting network.
  -debug             Set loglevel for TensorFlow and Logger to debug.
  -best              Just the best results of the networks will be used in -train or -plot
  -plot              Plots the last results of the specified networks in one file.
  -seg_size          The segment size used during training and testing
  -vec_size          The vector size to be expected when choosing a certain out_layer
  -out_layer         The layer index of the network to use for testing / creating the embeddings
  -epoch_total       number of epochs to train the network
  -epochs_pre_alr    number of epochs to train before active learning process starts
  -alr               number of active learning rounds (can be 0)
  -dense_factor      width of certain dense layers can be controlled via this parameter
  -output_size       width of the last dense layer (should be >= dense_factor)
```
  
As an example, you want to train, and test but not plot the network pairwise_lstm. You would call:
> controller.py -n pairwise_lstm -train -test

### General remarks
Before you start with your training you must run the controller once with the setup flag. 
This can take a while, as it extracts the speaker spectrograms from your audio files from all speakers specified in the common/extrapolation/speaker_factory.py class.
> controller.py -setup

### Folder structure, plots, results
* Trained network models will be stored in common/data/experiments/nets.
* Testresults will be stored in common/data/experiments/results. There are intermediate checkpoint files in respective subfolders allowing the process to abort and resume at the last possible checkpoint. These intermediate checkpoints are cleared once the process is completed entirely.
* Plots are stored in common/data/experiments/plots.
* For each network the implementation code is located below networks/&lt;name of net&gt;/

### Network identification
To support running various experiments of the same network, files (e.g. models and plots) contain additional identification information. This identifier may consist of parameters you used to start the experiment. Check the {{get_network_name()}} and {{get_formatted_result_network_name()}} functions if derived or the base implementation in network_controller.py.