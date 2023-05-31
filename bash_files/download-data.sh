#!/usr/bin/bash

cur_dir=$(pwd)
ego4d --output_directory=$cur_dir/videos/ --datasets clips --video_uid_file $cur_dir/sampled_clip_uids.txt -y 

VQ2D_ROOT="/media/goku/4b66c306-b38b-4701-9bd5-fd5c65a905fd/asjad.s/EGO4D/vq2d_cvpr"
mkdir -p $cur_dir/data/clips
mkdir -p $VQ2D_ROOT/data/clips
cp $cur_dir/videos/v2/clips/*.mp4 $VQ2D_ROOT/data/clips/ #$cur_dir/data/clips/

#rm -r videos

#mkdir -p $cur_dir/data/vq_splits
mkdir -p $VQ2D_ROOT/data/vq_splits


cd $VQ2D_ROOT

python process_vq_dataset.py --annot-root "$VQ2D_ROOT/data" --save-root "$VQ2D_ROOT/data/vq_splits" 
#python process_vq_dataset.py --annot-root "$cur_dir/data" --save-root "$cur_dir/data/vq_splits"

cd $cur_dir
