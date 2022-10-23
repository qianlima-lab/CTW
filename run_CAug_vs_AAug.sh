#! /bin/bash

#CAug
nohup python ./src/main.py --model single_ae_aug_after_sel --epochs 300 --lr 1e-3 --label_noise 0 \
--embedding_size 32 --ni 0.2 --mean_loss_len 5 --num_workers 2 --gamma 0.3 --aug GNoise \
--add_noise_interval 3 --cuda_device 0 --outfile CAug_GN.csv >after_GNoise.out 2>&1 &

nohup python ./src/main.py --model single_ae_aug_after_sel --epochs 300 --lr 1e-3 --label_noise 0 \
--embedding_size 32 --ni 0.2 --mean_loss_len 5 --num_workers 2 --gamma 0.3 --aug Oversample \
--add_noise_interval 3 --cuda_device 0 --outfile CAug_Oversample.csv >after_Oversample.out 2>&1 &

nohup python ./src/main.py --model single_ae_aug_after_sel --epochs 300 --lr 1e-3 --label_noise 0 \
--embedding_size 32 --ni 0.2 --mean_loss_len 5 --num_workers 2 --gamma 0.3 --aug Convolve \
--add_noise_interval 3 --cuda_device 0 --outfile CAug_Convolve.csv >after_Convolve.out 2>&1 &

nohup python ./src/main.py --model single_ae_aug_after_sel --epochs 300 --lr 1e-3 --label_noise 0 \
--embedding_size 32 --ni 0.2 --mean_loss_len 5 --num_workers 2 --gamma 0.3 --aug Crop \
--add_noise_interval 3 --cuda_device 0 --outfile CAug_Crop.csv >after_Crop.out 2>&1 &

nohup python ./src/main.py --model single_ae_aug_after_sel --epochs 300 --lr 1e-3 --label_noise 0 \
--embedding_size 32 --ni 0.2 --mean_loss_len 5 --num_workers 2 --gamma 0.3 --aug Drift \
--add_noise_interval 3 --cuda_device 0 --outfile CAug_Drift.csv >after_Drift.out 2>&1 &

nohup python ./src/main.py --model single_ae_aug_after_sel --epochs 300 --lr 1e-3 --label_noise 0 \
--embedding_size 32 --ni 0.2 --mean_loss_len 5 --num_workers 2 --gamma 0.3 --aug TimeWarp \
--add_noise_interval 3 --cuda_device 0 --outfile CAug_TimeWarp.csv >after_TimeWarp.out 2>&1 &

nohup python ./src/main.py --model single_ae_aug_after_sel --epochs 300 --lr 1e-3 --label_noise 0 \
--embedding_size 32 --ni 0.2 --mean_loss_len 5 --num_workers 2 --gamma 0.3 --aug Mixup \
--add_noise_interval 3 --cuda_device 0 --outfile CAug_Mixup.csv >after_Mixup.out 2>&1 &

# AAug
nohup python ./src/main.py --model single_ae_aug_before_sel --epochs 300 --lr 1e-3 --label_noise 0 \
--embedding_size 32 --ni 0.2 --mean_loss_len 5 --num_workers 2 --gamma 0.3 --aug GNoise \
--add_noise_interval 3 --cuda_device 0 --outfile CAug_GN.csv >after_GNoise.out 2>&1 &

nohup python ./src/main.py --model single_ae_aug_before_sel --epochs 300 --lr 1e-3 --label_noise 0 \
--embedding_size 32 --ni 0.2 --mean_loss_len 5 --num_workers 2 --gamma 0.3 --aug Oversample \
--add_noise_interval 3 --cuda_device 0 --outfile CAug_Oversample.csv >after_Oversample.out 2>&1 &

nohup python ./src/main.py --model single_ae_aug_before_sel --epochs 1 --lr 1e-3 --label_noise 0 \
--embedding_size 32 --ni 0.2 --mean_loss_len 5 --num_workers 2 --gamma 0.3 --aug Convolve \
--add_noise_interval 3 --cuda_device 0 --outfile CAug_Convolve.csv >after_Convolve.out 2>&1 &

nohup python ./src/main.py --model single_ae_aug_before_sel --epochs 300 --lr 1e-3 --label_noise 0 \
--embedding_size 32 --ni 0.2 --mean_loss_len 5 --num_workers 2 --gamma 0.3 --aug Crop \
--add_noise_interval 3 --cuda_device 0 --outfile CAug_Crop.csv >after_Crop.out 2>&1 &

nohup python ./src/main.py --model single_ae_aug_before_sel --epochs 300 --lr 1e-3 --label_noise 0 \
--embedding_size 32 --ni 0.2 --mean_loss_len 5 --num_workers 2 --gamma 0.3 --aug Drift \
--add_noise_interval 3 --cuda_device 0 --outfile CAug_Drift.csv >after_Drift.out 2>&1 &

nohup python ./src/main.py --model single_ae_aug_before_sel --epochs 300 --lr 1e-3 --label_noise 0 \
--embedding_size 32 --ni 0.2 --mean_loss_len 5 --num_workers 2 --gamma 0.3 --aug TimeWarp \
--add_noise_interval 3 --cuda_device 0 --outfile CAug_TimeWarp.csv >after_TimeWarp.out 2>&1 &

nohup python ./src/main.py --model single_ae_aug_before_sel --epochs 300 --lr 1e-3 --label_noise 0 \
--embedding_size 32 --ni 0.2 --mean_loss_len 5 --num_workers 2 --gamma 0.3 --aug Mixup \
--add_noise_interval 3 --cuda_device 0 --outfile CAug_Mixup.csv >after_Mixup.out 2>&1 &
