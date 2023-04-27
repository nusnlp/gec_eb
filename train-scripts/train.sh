gpu=0
MODEL_DIR=$1
DATA_PATH=$2
FAIRSEQ_DIR=fairseq
MAX_TOKENS=4096
PSEUDO_PATH=$3

CUDA_VISIBLE_DEVICES=$gpu python -u $FAIRSEQ_DIR/train.py $DATA_PATH \
    --save-dir $MODEL_DIR \
    --arch transformer_vaswani_wmt_en_de_big \
    --max-tokens $MAX_TOKENS \
    --task translation \
    --restore-file $PSEUDO_PATH \
    --lr-scheduler inverse_sqrt \
    --lr 3e-4 \
    --warmup-updates 4000 \
    --optimizer adam \
    --lr 0.00003 \
    -s src \
    -t tgt \
    --adam-betas '(0.9,0.98)' \
    --dropout 0.3 \
    --clip-norm 1.0 \
    --criterion label_smoothed_cross_entropy \
    --label-smoothing 0.1 \
    --log-format simple \
    --reset-lr-scheduler \
    --reset-optimizer \
    --reset-meters \
    --reset-dataloader \
    --max-epoch 10 \
    --seed 1 \
    --skip-invalid-size-inputs-valid-test
