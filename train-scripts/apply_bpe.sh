SUBWORD_NMT=subword-nmt
DATA_DIR=$1
PROCESSED_DIR="$DATA_DIR-data"
train_src=$DATA_DIR/train.src
train_tgt=$DATA_DIR/train.tgt
valid_src=$DATA_DIR/valid.src
valid_tgt=$DATA_DIR/valid.tgt

$SUBWORD_NMT/apply_bpe.py -c bpe/dict.bpe10000.txt < $train_src > $PROCESSED_DIR/train.src
$SUBWORD_NMT/apply_bpe.py -c bpe/dict.bpe10000.txt < $train_tgt > $PROCESSED_DIR/train.tgt
$SUBWORD_NMT/apply_bpe.py -c bpe/dict.bpe10000.txt < $valid_src > $PROCESSED_DIR/valid.src
$SUBWORD_NMT/apply_bpe.py -c bpe/dict.bpe10000.txt < $valid_tgt > $PROCESSED_DIR/valid.tgt
