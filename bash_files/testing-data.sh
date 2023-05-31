#!/usr/bin/bash


content="/media/goku/4b66c306-b38b-4701-9bd5-fd5c65a905fd/asjad.s/EGO4D"
cur_dir=$(pwd)

export LD_LIBRARY_PATH=home/goku/anaconda3/envs/asjad/lib:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=/usr/local/lib:$LD_LIBRARY_PATH

export PATH=/home/goku/anaconda3/envs/asjad/bin/:$PATH
export PATH=/home/goku/anaconda3/envs/asjad/bin/nvcc:$PATH
export PATH=/home/goku/anaconda3/envs/asjad/:$PATH

ego4d --output_directory="$content/ego4d_data/" --datasets vq2d_models -y
# Unzip
cd $content/ego4d_data/v2/vq2d_models

########################UNCOMMENT THIS FOR NEW PC########################

#unzip vq2d_models.zip 

#########################################################################

mkdir -p /$content/experiments/$1/logs
mkdir -p /$content/experiments/$1/visual_queries_logs

#cp $content/ego4d_data/v2/vq2d_models/pretrained_models/improved_baselines/* $content/experiments/$1/logs/
cp $content/ego4d_data/v2/vq2d_models/pretrained_models/siam_rcnn_residual/* $content/experiments/$1/logs/
cp $content/ego4d_data/v1_0_5/annotations/vq_test_unannotated.json $content/vq2d_cvpr/data/vq_splits/
cp $content/ego4d_data/v1_0_5/annotations/vq_train.json $content/vq2d_cvpr/data/vq_splits/
cp $content/ego4d_data/v1_0_5/annotations/vq_val.json $content/vq2d_cvpr/data/vq_splits/


cd $cur_dir
python setup_yaml.py

#cd $content/ego4d_data/v2/vq2d_models

VQ2D_ROOT="/media/goku/4b66c306-b38b-4701-9bd5-fd5c65a905fd/asjad.s/EGO4D/vq2d_cvpr"
cd $VQ2D_ROOT

EXPT_ROOT="$content/experiments/$1"
PYTRACKING_ROOT="$content/vq2d_cvpr/dependencies/pytracking"
VQ2D_SPLITS_ROOT="$content/vq2d_cvpr/data/vq_splits"
CLIPS_ROOT="$content/vq2d_cvpr/data/clips"

#source activate pytorch_env

PYTHONPATH=$VQ2D_ROOT:$PYTHONPATH
PYTHONPATH=$PYTRACKING_ROOT:$PYTHONPATH
PYTRACKING_ROOT="$VQ2D_ROOT/dependencies/pytracking"

export PYTHONPATH

python evaluate_vq2d.py \
  data.data_root="$CLIPS_ROOT" \
  data.split="val" \
  data.annot_root="$VQ2D_SPLITS_ROOT" \
  data.num_processes=1 \
  model.config_path="$EXPT_ROOT/logs/config.yaml" \
  model.checkpoint_path="$EXPT_ROOT/logs/model.pth" \
  logging.save_dir="$EXPT_ROOT/visual_queries_logs" \
  logging.stats_save_path="$EXPT_ROOT/visual_queries_logs/vq_stats.json.gz"


cd $cur_dir
python get_predictions.py "$EXPT_ROOT/visual_queries_logs/vq_stats.json.gz"
