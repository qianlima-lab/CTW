#!/bin/bash

############################################# Our Model: CTW ##############################################
nohup python ./src/main.py --model CTW --epochs 300 --lr 1e-3 --label_noise 0 \
--embedding_size 32 --ni 0.3 --num_workers 1 --add_noise_interval 1 --mean_loss_len 10 --gamma 0.3 \
--cuda_device 0 --aug TimeWarp --auto_rate 2 --outfile CTW.csv >CTW_sym30.out 2>&1 &
###########################################################################################################

nohup python ./src/main.py --model vanilla --epochs 300 --lr 1e-3 --label_noise 0 \
--embedding_size 32 --ni 0.3 --num_workers 1 --cuda_device 0 --outfile vanilla.csv >ucr0_vanilla_sym30.out 2>&1 &

nohup python ./src/main.py --model co_teaching_mloss --epochs 300 --lr 1e-3 --label_noise 0 \
--embedding_size 32 --ni 0.3 --mean_loss_len 10 --num_workers 2 --gamma 0.3 \
--cuda_device 0 --aug NoAug --outfile co_teaching.csv > ucr0co_teaching_mloss_sym30.out 2>&1 &

nohup python ./src/main.py --model sigua --epochs 300 --lr 1e-3 --label_noise 0 \
--embedding_size 32 --ni 0.3 --num_workers 1 --cuda_device 0 --ucr 0 --outfile sigua.csv >ucr0sigua_sym30.out 2>&1 &

nohup python ./src/SREA_single_experiment.py --epochs 300 --learning_rate 1e-3 --label_noise 0 --M 60 120 180 240 --delta_start 30 \
--delta_end 90 --embedding_size 32 --ni 0.3 --num_workers 1 --cuda_device 0 --outfile SREA.csv >ucr0_SREA_sym30.out 2>&1 &

nohup python ./src/BMM_single_experiment.py --model BMM --epochs 300 --lr 1e-3 --label_noise 0 \
--embedding_size 32 --ni 0.3 --num_workers 1 --cuda_device 0 --ucr 0 --outfile BMM.csv >BMM_sym30.out 2>&1 &

nohup python ./src/main.py --model dividemix --epochs 300 --lr 1e-3 --label_noise 0 --alpha 4 --lambda_u 25 \
--embedding_size 32 --ni 0.3 --num_workers 1 --cuda_device 0 --ucr 0 --outfile dividemix.csv >dividemix_sym30.out 2>&1 &

nohup python ./src/Sel_CL_experiment.py --low_dim 128 --lr-scheduler step --lr 0.001 --wd 1e-4 --download True --network FCN \
--embedding_size 512 --sup_t 0.1 --headType Linear --sup_queue_use 1 --sup_queue_begin 3 --queue_per_class 1000 \
--alpha 0.5 --beta 0.25 --k_val 250 --experiment_name UCR --alpha_m 1.0 --seed_initialization 1 --seed_dataset 42 \
--uns_t 0.1 --uns_queue_k 10000 --lr-warmup-epoch 5 --warmup-epoch 1 --lambda_s 0.01 --lambda_c 1 --warmup_way uns \
--num_workers 2 --ucr 0 --ft_lr 0.0001 --ni 0.3 --label_noise 0 --noise_type symmetric --cuda_device 0 --fine_tune \
--out ./ft_sym --epoch 230 --ft_epoch 70 --outfile Sel_CL.csv >sym30_sel_cl_ft.out 2>&1 &