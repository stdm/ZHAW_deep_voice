#!/bin/sh
DEFAULT_NAME="benchmarking_speakerclustering-latest"
read -p "Enter name of the container: [$DEFAULT_NAME] " NAME
NAME="${NAME:-$DEFAULT_NAME}"
if [ -f ./$NAME.simg ]; then
	 srun --pty --ntasks=1 --cpus-per-task=4 --mem=16G --qos=student-normal --gres=gpu:1 singularity shell $NAME.simg
else
	echo "File $NAME.simg does not exist."
fi
