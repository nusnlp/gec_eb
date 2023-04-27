cd to/directory/which/contains/fairseq

gpu=1
MODEL_DIR=$1
DATA_PATH=$2
actor=embedding
FAIRSEQ_DIR=fairseq
MAX_TOKENS=9000
PSEUDO_PATH=$3

CUDA_VISIBLE_DEVICES=$gpu python -u $FAIRSEQ_DIR/train.py $DATA_PATH \
    --save-dir $MODEL_DIR \
    --num-sources 7 \
    --arch transformer_vaswani_wmt_en_de_big \
    --max-tokens $MAX_TOKENS \
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
    --max-epoch 1 \
    --seed 1 \
    --update-weight 280 \
    --data-actor-path $MODEL_DIR \
    --skip-invalid-size-inputs-valid-test \
    --data-actor embedding \
    --data-actor-lr 0.0001 \
    --nos-data-actor-lr 0.0001 \
    --data-actor-optim-step 100 \
    --hard-sample \
    --just-two \
    --seed 1
