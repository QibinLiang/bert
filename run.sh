#!/bin/bash

num_gpus=4
ddp_file=checkpoints/ddp_init
for((i=0; i<$num_gpus; ++i)); do
{
  gpu_id=$i
  init_file=file://$(readlink -f $ddp_file)
  echo running on rank "$gpu_id"
  python train.py --rank "$gpu_id" --init_file "$init_file"
} &
done
wait
