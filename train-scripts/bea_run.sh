cd /home/hannan/workspace/working/exposure-bias-debug

. $HOME/miniconda3/etc/profile.d/conda.sh
conda activate /home/hannan/miniconda3/envs/debug_eb

gpu=$1
trade_off=0.5
MODEL_DIR=$2
DATA_PATH=data/WI-sample/pretrain.ckpt=8.clang8+fce+nucle-wi/WI-5-data-bin
FAIRSEQ_DIR=fairseq
BATCH=80
PSEUDO_PATH=/home/hannan/workspace/reproduce/c4_200m/training-code/model/train/pretrain-ckpt-8-clang8+fce+nucle-wi/checkpoint1.pt

CUDA_VISIBLE_DEVICES=$gpu python -u $FAIRSEQ_DIR/train.py $DATA_PATH \
    --save-dir $MODEL_DIR \
    --num-sources 7 \
    --arch transformer_vaswani_wmt_en_de_big \
    --max-tokens 9200 \
    --task reweighting \
    --restore-file $PSEUDO_PATH \
    --lr-scheduler inverse_sqrt \
    --optimizer adam \
    --lr 0.00003 \
    -s src \
    -t tgt \
    --dropout 0.3 \
    --clip-norm 1.0 \
    --criterion reweight_modified \
    --label-smoothing 0.1 \
    --adam-betas '(0.9,0.98)' \
    --log-format simple \
    --reset-lr-scheduler \
    --reset-optimizer \
    --reset-meters \
    --reset-dataloader \
    --max-epoch 20 \
    --seed 1 \
    --update-weight 220 \
    --data-actor-path $MODEL_DIR \
    --skip-invalid-size-inputs-valid-test \
    --data-actor embedding \
    --data-actor-lr 0.0001 \
    --nos-data-actor-lr 0.0001 \
    --nos-data-actor-optim-step 100 \
    --hard-sample \
    --just-two
