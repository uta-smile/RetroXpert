dataset=USPTO50K-aug-typed

python  $PWD/../../onmt/bin/preprocess.py \
    -train_src $PWD/../../data/${dataset}/src-train-aug-err.txt \
    -train_tgt  $PWD/../../data/${dataset}/tgt-train-aug-err.txt \
    --valid_src  $PWD/../../data/${dataset}/src-valid.txt \
    --valid_tgt  $PWD/../../data/${dataset}/tgt-valid.txt \
    -save_data  $PWD/../../data/${dataset}/${dataset} \
    --src_seq_length 1000 --tgt_seq_length 1000 \
    --src_vocab_size 1000 --tgt_vocab_size 1000 --share_vocab  --overwrite
