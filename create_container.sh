#!/bin/sh
read -p "Enter dockerhub username: " USERNAME
read -p  "Enter dockerhub password: " PASSWORD
export SINGULARITY_DOCKER_USERNAME=$USERNAME
export SINGULARITY_DOCKER_PASSWORD=$PASSWORD
singularity pull docker://papommerman/benchmarking_speakerclustering:latest

