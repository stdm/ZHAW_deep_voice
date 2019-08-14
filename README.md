# ZHAW deep voice
 
 The ZHAW deep voice is a package of multiple neural networks that try resolving the speaker clustering task. The goal is to provide a uniform way of data-access, -preprocession and analysis fo the results.
 
## Data sets
The suite supports two datasets at this point. The TIMIT dataset and the VoxCeleb2 dataset.
Further datasets may be added. To do so, you need to edit the `common/extrapolation/speaker.py` and `common/extrapolation/speaker_factory.py` to specify handling of dataset specifics.
The speaker list files are located in `common/data/speaker_lists`.

### TIMIT
This is a paid product from the LDC and can be [obtained here](https://www.ldc.upenn.edu/).
This data also needs to be processed using the [sph2pipe tool](https://www.ldc.upenn.edu/language-resources/tools/sphere-conversion-tools) and be put in the folder `common/data/training/TIMIT`.

### VoxCeleb2
This is a freely available dataset set and can be [obtained here](http://www.robots.ox.ac.uk/~vgg/data/voxceleb/vox2.html). It contains m4a-format audiofiles which need to be converted to .wav files to enable processing. You may use the audio_converter.py script to do so or alternatively use a tool of your choice (e.g. [FFmpeg](https://ffmpeg.org/)).
The converted data needs to be put in the folder `common/data/training/VOXCELEB2`.

## Using deep voice
If you simply want to use it, you can let docker do the work for you and let it import all needed packages.

In any way, whether you fork and pull the source code or let docker handle it for you, the whole suite is controllable over a one file interface,  `controller.py`.

 To view all available parametrs call
 ```
 controller.py --help
 ```

Arguments are also listed here: 
```
controller.py
  -h, --help         show this help message and exit
  -setup             Run project setup.
  -n NETWORKS        The networks to use for training or analysis. Available: 'pairwise_lstm', 'pairwise_lstm_vox2', 'arc_face', 'pairwise_kldiv', 'luvo', 'gmm', 'i-vector'. All networks use different sets of parameters which can be configured in the file common/config.cfg
  -train             Train the specified network.
  -test              Test the specified network.
  -best              Just the best results of the networks will be used in -train or -plot
  -plot              Plots the last results of the specified networks in one file.
  -dev               Enable dev mode so the dev set instead of the test set is used for testing.
  -config            The config file to use for training or analysis
```
  
As an example, you want to test and plot but not train the networks pairwise_kldiv and pairwise_lstm. You would call:
> controller.py -n pairwise_kldiv pairwise_lstm -test -plot

### General remarks
Before you start with your training you must run the controller once with the setup flag. 
This can take a while, as it extracts the speaker spectrograms from your audio files from all speakers specified in the `common/extrapolation/speaker_factory.py` class.
> controller.py -setup

### Folder structure, plots, results
* Trained network models will be stored in `common/data/experiments/nets`.
* Test results will be stored in `common/data/experiments/results`. There are intermediate checkpoint files in respective subfolders allowing the process to abort and resume at the last possible checkpoint. These intermediate checkpoints are cleared once the process is completed entirely.
* Plots are stored in `common/data/experiments/plots`.
* The name of the trained models, results and plots contains the name of the net and the config file used.
* For each network the implementation code is located below `networks/<name of net>`
