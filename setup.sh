#!/bin/bash
pwd
mkdir -p /content/data/
wget https://dl.fbaipublicfiles.com/avhubert/model/lrs3_vox/vsr/base_vox_433h.pt -O /content/data/finetune-model.pt
git clone https://github.com/facebookresearch/av_hubert.git
cd av_hubert
pip install -r requirements.txt
pip install --editable ./fairseq
cd ..
pip install jiwer

