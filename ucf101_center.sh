#!/bin/bash

#SBATCH --job-name example_selection_test
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-gpu=12
#SBATCH --mem-per-gpu=30G
#SBATCH --time 1-00:00:0
#SBATCH --partition batch_ce_ugrad
#SBATCH -w moana-y3
#SBATCH -o results/ucf101_center_frame_place365/%A-%x.out
#SBATCH -e results/ucf101_center_frame_place365/%A-%x.err




python /data/jong980812/project/Video-CBM/train_video_cbm.py \
--dataset ucf101 \
--data_set UCF101 \
--nb_classes 101 \
--concept_set /data/jong980812/project/Video-CBM/data/concept_sets/places365_filtered.txt \
--batch_size 24 \
--video-anno-path data/video_annotation/ucf101 \
--data_path /local_datasets/ucf101/videos \
--print \
--center_frame \
--activation_dir /data/jong980812/project/Video-CBM/results/ucf101_center_frame_place365/activation \
--save_dir /data/jong980812/project/Video-CBM/results/ucf101_center_frame_place365/model \
--proj_steps 2000 \
--n_iters 2000 \
--interpretability_cutoff 0.55