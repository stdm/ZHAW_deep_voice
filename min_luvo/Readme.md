This is a minimal working luvo controller that can be trained.

The original version has a memory leak in one of its dependencies. This version only contains the neccessary code to be trained. The dockerfile contains all necessary dependencies and is right now working.

To run it, simply copy the speakers_590_clustering_without_raynolds_train.pickle from the original code into this subdirectory and run LuVo.py from this directory inside the docker container, that can be built from the Dockerfile in this directory. The memory leak can be provoked by simply running the LuVo.py file inside the docker container that is built from the original Dockerfile.

This directory is only for locating the memory leak and will be merged with the original code.
