GEC-EB: Mitigating Exposure Bias in Grammatical Error Correction with Data Augmentation and Reweighting

> Hannan Cao, Wenmian Yang, Hwee Tou Ng. Mitigating Exposure Bias in Grammatical Error Correction with Data Augmentation and Reweighting. In EACL 2023. 

The program is tested under pytorch 1.7.1, CUDA version 11.7 

1. Download required data and install required software

	1.1. Generate the C4 200M synthetic data by following https://github.com/google-research-datasets/C4_200M-synthetic-dataset-for-grammatical-error-correction
	
	1.2. Download the NUCLE from https://sterling8.d2.comp.nus.edu.sg/nucle_download/nucle.php ; FCE from https://www.cl.cam.ac.uk/research/nl/bea2019st/data/fce_v2.1.bea19.tar.gz ; CLang8 from https://github.com/google-research-datasets/clang8 ; Download W&I dataset from https://www.cl.cam.ac.uk/research/nl/bea2019st/ ; Downlaod CoNLL-2013 dataset from  https://www.comp.nus.edu.sg/~nlp/conll13st/release2.3.1.tar.gz and CoNLL-2014 dataset from https://www.comp.nus.edu.sg/~nlp/conll14st/conll14st-test-data.tar.gz ; Download CWEB dataset from https://github.com/SimonHFL/CWEB
	
	1.3. Install the fairseq inside the fairseq folder
	```
	cd fairseq
	pip3 install --editable ./
	```
Note all the scripts are inside train-scripts folder.

2. Pretrain and Train the Transformer-big model
	2.1 Pass the path for target sentences into tok+bpe+pre.sh to generate the bpe using the subword_nmt package (https://github.com/rsennrich/subword-nmt)
	```
	./tok+bpe+pre.sh
	```
	2.2 Use apply_bpe.sh and create-dict-preprocess.sh to preprocess the pre-training data. 
	```
	./apple_bpe.sh path/to/pretrain/data/folder
	./create-dict-preprocess.sh path/to/bpe-ed/data/folder
	```
	2.3 Pretrain the model with pretrain.sh
	```
	./pretrain.sh
	```
	2.4 Preprocess the training data 
	```
	./apple_bpe.sh path/to/train/data/folder
	./preprocess.sh path/to/bpe-ed/data/folder
	```
	2.5 Train the model with train.sh
	```
	./train.sh model/train preprocessed/train/data path/to/pretrained/checkpoint
	```
3. Generate augmented sentence
	3.1. Use downloaded checkpoint to make predicitions on the training set (need to specify):
	```
	./predict.sh 0 path/to/source/training/sentence "candidate_data" path/to/downloaded/checkpoint output/directory
	```
	3.2. Generate candidate sentences from the prediction result, move the candidate files to respective folders (e.g. neg-1, neg-2, 	neg-3, neg-4, neg-5 are the respective folders and assume original training and validation sentences are stored in pos folder):
	```
	python generate_candidates.py --root_path previous/used/output/directory --candidate_name test.nbest.tok.candidate_data
	mkdir pos
	mkdir neg-1
	mkdir neg-2
	mkdir neg-3
	mkdir neg-4
	mkdir neg-5
	mkdir pos-data
	mkdir neg-1-data
	mkdir neg-2-data
	mkdir neg-3-data
	mkdir neg-4-data
	mkdir neg-5-data
	mv candi.1 neg-1/train.tgt
	mv candi.2 neg-2/train.tgt
	mv candi.3 neg-3/train.tgt
	mv candi.4 neg-4/train.tgt
	mv candi.5 neg-5/train.tgt
	```
	3.3. Copy the train.src, valid.src and valid.tgt to neg-1, neg-2, neg-3, neg-4, neg-5 folders
	3.4. Create the count for the number of candidates:
	```
	python valid_count.py --candi_path path/to/your/source/training/data/folder --file_name output/file/name --count numner/of/candidates/you/selected
	```
	3.5. Pass neg-1, neg-2, ..., neg-5 folders to apply_bpe.py to process the data
	```
	./apply_bpe.sh /path/to/neg-1/folder
	```
	3.6. Combine augmented data together (e.g. combine 5 together):
	```
	python multi_target.py --source /path/to/processed/pos/train.tgt/file /path/to/processed/neg-1/train.tgt/file ... /path/to/processed neg-5/train.tgt/file /path/to/count/file" \
						--target /path/to/processed/pos/train.src/file \
						--max 750 \
						--ratio 750 \
						--out path/to/output/data/folder
	```
	3.7. Binarize the data using preprocess.sh
	```
	./preprocess.sh path/to/output/data/folder
	```
4. Train the model using DM method. 
```
./conll_run.sh 0 path/to/save/finetuned/checkpoint
./bea_run.sh 0 path/to/save/finetuned/checkpoint
```
5. Make prediction with predict.sh
```
./predict.sh 0 path/to/test/set randome/name path/to/finetuned/weight output/directory
```
6. Use M2 scorer to evaluate the result of CoNLL-2014 test set, and evaluate the result on BEA-2019 test set by submitting the prediction result to colab:https://competitions.codalab.org/competitions/20228#participate-get-data


## Citation

If you found our paper or code useful, please cite as:

```
@inproceedings{cao-etal-2022-eb,
    title = "Mitigating Exposure Bias in Grammatical Error Correction with Data Augmentation and Reweighting",
    author = "Cao, Hannan  and
      Yang, Wenmian  and
      Ng, Hwee Tou",
    booktitle = "Proceedings of the 17th Conference of the European Chapter of the Association for Computational Linguistics",
    year = "2023",
}
```

## License
The source code and models in this repository are licensed under the GNU General Public License Version 3 (see [License](./LICENSE.txt)). For commercial use of this code and models, separate commercial licensing is also available. Please contact Hwee Tou Ng (nght@comp.nus.edu.sg)
