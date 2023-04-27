gpu=0,1
MODEL_DIR=model/pretrain
DATA_PATH=pretrain-data-10000-bin
FAIRSEQ_DIR=fairseq
MAX_TOKENS=16000
UPDATE_FRQ=1
SEED=1

CUDA_VISIBLE_DEVICES=$gpu python -u $FAIRSEQ_DIR/train.py $DATA_PATH \
    --save-dir $MODEL_DIR \
    --arch transformer_vaswani_wmt_en_de_big \
    --max-tokens $MAX_TOKENS \
    --optimizer adam \
    --adam-betas '(0.9,0.98)' \
    --lr-scheduler inverse_sqrt \
    --lr 3e-4 \
    --warmup-updates 8000 \
    --warmup-init-lr 1e-7 \
    --min-lr 1e-9 \
    -s src \
    -t tgt \
    --dropout 0.3 \
    --clip-norm 1.0 \
    --criterion label_smoothed_cross_entropy \
    --label-smoothing 0.1 \
    --log-format simple \
    --max-epoch 10 \
    --seed $SEED \
    --update-freq $UPDATE_FRQ \
    --skip-invalid-size-inputs-valid-test  
