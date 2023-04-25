#!/bin/bash

# add '--ucr 128' for training all datasets in UCR
# --label_noise 0: symmetric noise
# --label_noise 1: asymmetric noise
# --label_noise -1: instance-depended noise
############################################# Our Model: CTW ##############################################
nohup python ./src/main.py --model CTW --epochs 300 --lr 1e-3 --label_noise 0 \
--embedding_size 32 --ni 0.3 --num_workers 1 --arg_interval 1 --mean_loss_len 10 --gamma 0.3 \
--cuda_device 0 --outfile CTW.csv >/dev/null 2>&1 &
###########################################################################################################

# Vanilla
nohup python ./src/main.py --model vanilla --epochs 300 --lr 1e-3 --label_noise 0 \
--embedding_size 32 --ni 0.3 --num_workers 1 --cuda_device 0 --outfile vanilla.csv >/dev/null 2>&1 &

# Co-teaching
nohup python ./src/main.py --model co_teaching_mloss --epochs 300 --lr 1e-3 --label_noise 1 \
--embedding_size 32 --ni 0.4 --mean_loss_len 10 --num_workers 2 --gamma 0.3 --ucr 128 \
--cuda_device 1 --aug NoAug --outfile co_teaching_asym40.csv >/dev/null 2>&1 &

# SIGUA
nohup python ./src/main.py --model sigua --epochs 300 --lr 1e-3 --label_noise 0 \
--embedding_size 32 --ni 0.3 --num_workers 1 --cuda_device 0 --ucr 0 --outfile sigua.csv >/dev/null 2>&1 &

# SREA
nohup python ./src/SREA_single_experiment.py --epochs 300 --learning_rate 1e-3 --label_noise 0 --M 60 120 180 240 --delta_start 30 \
--delta_end 90 --embedding_size 32 --ni 0.3 --num_workers 1 --cuda_device 0 --outfile SREA.csv >/dev/null 2>&1 &

# Mixup-BMM
nohup python ./src/BMM_single_experiment.py --model BMM --epochs 300 --lr 1e-3 --label_noise 0 \
--embedding_size 32 --ni 0.3 --num_workers 1 --cuda_device 0 --ucr 0 --outfile BMM.csv >/dev/null 2>&1 &

# DivideMix
nohup python ./src/main.py --model dividemix --epochs 300 --lr 1e-3 --label_noise 0 --alpha 4 --lambda_u 25 \
--embedding_size 32 --ni 0.3 --num_workers 1 --cuda_device 0 --ucr 0 --outfile dividemix.csv >/dev/null 2>&1 &

# Sel-CL
nohup python ./src/Sel_CL_experiment.py --low_dim 128 --lr-scheduler step --lr 0.001 --wd 1e-4 --download True --network FCN \
--embedding_size 512 --sup_t 0.1 --headType Linear --sup_queue_use 1 --sup_queue_begin 3 \
--alpha 0.5 --beta 0.25 --experiment_name TimeSeries --alpha_m 1.0 --seed_initialization 1 --seed_dataset 42 \
--uns_t 0.1 --lr-warmup-epoch 5 --warmup-epoch 1 --lambda_s 0.01 --lambda_c 1 --warmup_way uns \
--num_workers 1 --ucr 0 --ft_lr 0.0001 --ni 0.3 --label_noise 0 --cuda_device 0 \
--out ./ft_sym --epoch 300 --outfile Sel_CL.csv >/dev/null 2>&1 &


