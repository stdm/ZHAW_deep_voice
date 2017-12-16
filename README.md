 # ZHAW deep voice
 
 The ZHAW deep voice is a package of multiple neural networks that try resolving the speaker clustering task. The goal is to provide a uniform way of data-access, -preprocession and analysis fo the results.
 
 Note that the suite needs the TIMIT Dataset to function at this point. This is a paid product from the LDC and can be [obtained here.](https://www.ldc.upenn.edu/)
This data also needs to be processed using the [sph2pipe tool](https://www.ldc.upenn.edu/language-resources/tools/sphere-conversion-tools) and be put in the folder common/data/training/TIMIT

 ## Using deep voice
 If you simply want to use it, you can let docker do the work for you and let it import all needed packages.
 
 In any way, whether you fork and pull the source code or let docker handle it for you, the whole suite is controllable over a one file interface, controller.py.
 It can be run from console with the following calling structure:
  controller.py [-h] [-setup] [-n network] [-train] [-test] [-plot] [-clear] [-debug] [-best] [-val# ne]
  - -help Display the help screen you are seeing here
  - -setup Create all 
  - -n specifiy which network should be used. Available:
  'pairwise_lstm', 'pairwise_kldiv', 'flow_me', 'luvo' and 'all' (without the single quotes)
  - -train Specify to train the chosen network
  - -test Specify to test the chosen network
  - -plot Specify to plot the results of the chosen network. If network is 'all', all results will be displayed in one single plot
  - -clear Clear the folder in experiments
  - -debug Set the logging level of Tensorflow to Debug
  - -best Just the best results of the networks will be used in -plot
  - -val# specify which speaker number you want to use (40, 60, 80) to test the networks
  
As an example, you want to train, and test but not plot the network pairwise_lstm. you would call:
> controller.py -n pairwise_lstm -train -test

### General remarks
Before you start with your training you should run the controller once with the setup flag. This can take a while, approximately around 10 minutes.
