#!/bin/bash
#CCS -N myJob
#CCS --res=rset=1:ncpus=1:mem=1g
#CCS -t 50m
#CCS -M mhshaker@mail.upb.de
#CCS -mea
#CCS -J 0-$2:1

# echo ${CCS_ARRAY_INDEX}
python3 Sampling.py $1 ${CCS_ARRAY_INDEX}

