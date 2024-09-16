#!/bin/bash
#SBATCH --job-name example_selection_test
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-gpu=12
#SBATCH --mem-per-gpu=50G
#SBATCH --time 1-00:00:0
#SBATCH --partition batch_ce_ugrad
#SBATCH -w moana-y5
#SBATCH -o /data/jong980812/project/Video-CBM/results/ucf101/class_list/%A-%x.out
#SBATCH -e /data/jong980812/project/Video-CBM/results/ucf101/class_list/%A-%x.err

OUTPUT_DIR='/data/jong980812/project/Video-CBM/results/ucf101/class_list'


python /data/jong980812/project/Video-CBM/train_video_cbm.py \
--data_set UCF101 \
--nb_classes 101 \
--concept_set /data/jong980812/project/Video-CBM/data/ucf101_classes.txt \
--batch_size 48 \
--video-anno-path data/video_annotation/ucf101 \
--data_path /local_datasets/ucf101/videos \
--print \
--activation_dir ${OUTPUT_DIR}/activation \
--save_dir ${OUTPUT_DIR}/model \
--proj_steps 2000 \
--n_iters 2000 \
--backbone vmae_vit_base_patch16_224 \
--finetune /data/datasets/video_checkpoint/ucf101/ucf_finetune.pth \
--lavila_ckpt /data/datasets/video_checkpoint/lavila_TSF_B.pth \
--dual_encoder internvid \
--clip_cutoff 0.1 \
--interpretability_cutoff 0.1 \