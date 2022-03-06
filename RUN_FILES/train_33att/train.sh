#!/bin/bash

#$-m abe
#$-M dyfdyf0125@gmail.com
#$-q gpu@qa-xp-001
#$-l gpu_card=4
#$-N CRC_ran_end2end

export PATH=/nfs/yding4/conda_envs/locate-and-label/bin:$PATH
export LD_LIBRARY_PATH=/nfs/yding4/conda_envs/locate-and-label/lib:$LD_LIBRARY_PATH

CODE_DIR=/nfs/yding4/AVE_project/sequence-to-set

CONFIG=/nfs/yding4/AVE_project/sequence-to-set/RUN_FILES/train_33att/33att.config

cd ${CODE_DIR}

/nfs/yding4/conda_envs/locate-and-label/bin/python ssn.py train    \
    --config    ${CONFIG}   \
    2>&1 | tee train.log


