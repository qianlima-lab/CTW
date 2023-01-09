#! /bin/bash

#CAug
nohup python ./src/main.py --model single_ae_aug_after_sel --epochs 300 --lr 1e-3 --label_noise 0 \
--embedding_size 32 --ni 0.2 --mean_loss_len 5 --num_workers 2 --gamma 0.3 --aug GNoise \
--arg_interval 3 --cuda_device 0 --outfile CAug_GN.csv >/dev/null 2>&1 &

nohup python ./src/main.py --model single_ae_aug_after_sel --epochs 300 --lr 1e-3 --label_noise 0 \
--embedding_size 32 --ni 0.2 --mean_loss_len 5 --num_workers 2 --gamma 0.3 --aug Oversample \
--arg_interval 3 --cuda_device 0 --outfile CAug_Oversample.csv >/dev/null 2>&1 &

nohup python ./src/main.py --model single_ae_aug_after_sel --epochs 300 --lr 1e-3 --label_noise 0 \
--embedding_size 32 --ni 0.2 --mean_loss_len 5 --num_workers 2 --gamma 0.3 --aug Convolve \
--arg_interval 3 --cuda_device 0 --outfile CAug_Convolve.csv >/dev/null 2>&1 &

nohup python ./src/main.py --model single_ae_aug_after_sel --epochs 300 --lr 1e-3 --label_noise 0 \
--embedding_size 32 --ni 0.2 --mean_loss_len 5 --num_workers 2 --gamma 0.3 --aug Crop \
--arg_interval 3 --cuda_device 0 --outfile CAug_Crop.csv >/dev/null 2>&1 &

nohup python ./src/main.py --model single_ae_aug_after_sel --epochs 300 --lr 1e-3 --label_noise 0 \
--embedding_size 32 --ni 0.2 --mean_loss_len 5 --num_workers 2 --gamma 0.3 --aug Drift \
--arg_interval 3 --cuda_device 0 --outfile CAug_Drift.csv >/dev/null 2>&1 &

nohup python ./src/main.py --model single_ae_aug_after_sel --epochs 300 --lr 1e-3 --label_noise 0 \
--embedding_size 32 --ni 0.2 --mean_loss_len 5 --num_workers 2 --gamma 0.3 --aug TimeWarp \
--arg_interval 3 --cuda_device 0 --outfile CAug_TimeWarp.csv >/dev/null 2>&1 &

nohup python ./src/main.py --model single_ae_aug_after_sel --epochs 300 --lr 1e-3 --label_noise 0 \
--embedding_size 32 --ni 0.2 --mean_loss_len 5 --num_workers 2 --gamma 0.3 --aug Mixup \
--arg_interval 3 --cuda_device 0 --outfile CAug_Mixup.csv >/dev/null 2>&1 &

# AAug
nohup python ./src/main.py --model single_ae_aug_before_sel --epochs 300 --lr 1e-3 --label_noise 0 \
--embedding_size 32 --ni 0.2 --mean_loss_len 5 --num_workers 2 --gamma 0.3 --aug GNoise \
--arg_interval 3 --cuda_device 0 --outfile CAug_GN.csv >/dev/null 2>&1 &

nohup python ./src/main.py --model single_ae_aug_before_sel --epochs 300 --lr 1e-3 --label_noise 0 \
--embedding_size 32 --ni 0.2 --mean_loss_len 5 --num_workers 2 --gamma 0.3 --aug Oversample \
--arg_interval 3 --cuda_device 0 --outfile CAug_Oversample.csv >/dev/null 2>&1 &

nohup python ./src/main.py --model single_ae_aug_before_sel --epochs 1 --lr 1e-3 --label_noise 0 \
--embedding_size 32 --ni 0.2 --mean_loss_len 5 --num_workers 2 --gamma 0.3 --aug Convolve \
--arg_interval 3 --cuda_device 0 --outfile CAug_Convolve.csv >/dev/null 2>&1 &

nohup python ./src/main.py --model single_ae_aug_before_sel --epochs 300 --lr 1e-3 --label_noise 0 \
--embedding_size 32 --ni 0.2 --mean_loss_len 5 --num_workers 2 --gamma 0.3 --aug Crop \
--arg_interval 3 --cuda_device 0 --outfile CAug_Crop.csv >/dev/null 2>&1 &

nohup python ./src/main.py --model single_ae_aug_before_sel --epochs 300 --lr 1e-3 --label_noise 0 \
--embedding_size 32 --ni 0.2 --mean_loss_len 5 --num_workers 2 --gamma 0.3 --aug Drift \
--arg_interval 3 --cuda_device 0 --outfile CAug_Drift.csv >/dev/null 2>&1 &

nohup python ./src/main.py --model single_ae_aug_before_sel --epochs 300 --lr 1e-3 --label_noise 0 \
--embedding_size 32 --ni 0.2 --mean_loss_len 5 --num_workers 2 --gamma 0.3 --aug TimeWarp \
--arg_interval 3 --cuda_device 0 --outfile CAug_TimeWarp.csv >/dev/null 2>&1 &

nohup python ./src/main.py --model single_ae_aug_before_sel --epochs 300 --lr 1e-3 --label_noise 0 \
--embedding_size 32 --ni 0.2 --mean_loss_len 5 --num_workers 2 --gamma 0.3 --aug Mixup \
--arg_interval 3 --cuda_device 0 --outfile CAug_Mixup.csv >/dev/null 2>&1 &
