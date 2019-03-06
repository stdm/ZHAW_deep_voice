#!/bin/bash
DOCKER_USER="papommerman"
DOCKER_REPO="benchmarking_speakerclustering"
TAG="latest"
read -p "Enter dockerhub username: " USERNAME
read -s -p "Enter dockerhub password: " PASSWORD
echo
export SINGULARITY_DOCKER_USERNAME=$USERNAME
export SINGULARITY_DOCKER_PASSWORD=$PASSWORD
if [ -f ./$DOCKER_REPO-$TAG.simg ]; then
	echo "File $DOCKER_REPO-$TAG already exists. Please delete it beforehand"
else
	singularity pull docker://$DOCKER_USER/$DOCKER_REPO:$TAG
fi
