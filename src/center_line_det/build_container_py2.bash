#!/bin/bash

CMD=$*

if [ -z "$CMD"];
then
	CMD=/bin/bash
fi

home_dir=./auto-lane-gen-experiments/src/center_line_det # the dir of CenterLineDet
dataset_dir=./auto-lane-gen-experiments/src/center_line_det/nuscenes # the dir of nuscenes dataset
container_name=centerlinedet_eval # container name 
port_number=5050 # port number for tensorboard

docker run -d \
	-v $home_dir
	-v $dataset_dir
	--name=$container_name\
	--gpus all\
	--shm-size 32G\
	-p $port_number:6006\
	# --rm -it zhxu_py2 $CMD

docker attach centerlinedet_eval
