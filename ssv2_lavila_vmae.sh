#!/bin/bash
#SBATCH --job-name spatio_origin_ssv2
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-gpu=12
#SBATCH --mem-per-gpu=30G
#SBATCH --time 1-00:00:0
#SBATCH --partition batch_ce_ugrad
#SBATCH -w moana-y3
#SBATCH -o /data/jong980812/project/Video-CBM/results/origin_ssv2/finetune_internvid_ost_spatio_temporal/%A-%x.out
#SBATCH -e /data/jong980812/project/Video-CBM/results/origin_ssv2/finetune_internvid_ost_spatio_temporal/%A-%x.err

OUTPUT_DIR='/data/jong980812/project/Video-CBM/results/origin_ssv2/finetune_internvid_ost_spatio_temporal'


python /data/jong980812/project/Video-CBM/train_video_cbm.py \
--data_set SSV2 \
--nb_classes 174 \
--concept_set /data/jong980812/project/Video-CBM/data/concept_sets/ssv2_ost_spatio_temporal_concepts.txt \
--batch_size 32 \
--video-anno-path data/video_annotation/ssv2 \
--data_path /local_datasets/something-something/something-something-v2-mp4 \
--activation_dir ${OUTPUT_DIR}/activation \
--save_dir ${OUTPUT_DIR}/model \
--proj_steps 2000 \
--n_iters 2000 \
--backbone vmae_vit_base_patch16_224 \
--finetune /data/datasets/video_checkpoint/ssv2/ssv2_finetune.pth \
--lavila_ckpt /data/datasets/video_checkpoint/lavila_TSF_B.pth \
--dual_encoder internvid \
--clip_cutoff 0.2