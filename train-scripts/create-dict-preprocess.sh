VOCAB_DIR=vocab
PROCESSED_DIR=$1
FAIRSEQ_DIR=fairseq
cpu_num=`grep -c ^processor /proc/cpuinfo`

python3 $FAIRSEQ_DIR/preprocess.py --source-lang src --target-lang tgt \
	--trainpref $PROCESSED_DIR/train \
	--validpref $PROCESSED_DIR/valid \
	--destdir "$PROCESSED_DIR-bin" \
	--workers $cpu_num \
	--thresholdsrc 50000 \
	--thresholdtgt 50000 \
	--joined-dictionary
