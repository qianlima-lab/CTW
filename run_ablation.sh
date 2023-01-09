#!/bin/bash

nohup python ./src/main.py --model single_ae_aug --epochs 300 --lr 1e-3 --label_noise 0 \
--embedding_size 32 --ni 0.3 --num_workers 2 --arg_interval 1 --aug TimeWarp \
--cuda_device 0 --outfile wo_sel.csv >/dev/null 2>&1 &

nohup python ./src/main.py --model single_ae_sel --epochs 300 --lr 1e-3 --label_noise 0 \
--embedding_size 32 --ni 0.3 --num_workers 2 --mean_loss_len 10 --gamma 0.3 \
--cuda_device 0 --sel_method 3 --outfile wo_TimeWarp.csv >/dev/null 2>&1 &

nohup python ./src/main.py --model single_aug_after_sel --epochs 300 --lr 1e-3 --label_noise 0 \
--embedding_size 32 --ni 0.3 --num_workers 2 --arg_interval 1 --aug TimeWarp --mean_loss_len 10 --gamma 0.3 \
--cuda_device 0 --sel_method 3 --outfile wo_decoder.csv >/dev/null 2>&1 &

nohup python ./src/main.py --model single_ae_aug_after_sel --epochs 300 --lr 1e-3 --label_noise 0 \
--embedding_size 32 --ni 0.3 --num_workers 1 --arg_interval 1 --mean_loss_len 10 --gamma 0.3 \
--cuda_device 0 --aug TimeWarp --sel_method 4 --outfile wo_EPS.csv >/dev/null 2>&1 &
