dataset=USPTO50K-aug-typed

python  ../../onmt/bin/train.py --data ../../data/${dataset}/${dataset} \
    --save_model ../../experiments/checkpoints/${dataset}/${dataset}_model \
    --seed 42 --world_size 2 --gpu_ranks 0 1 --save_checkpoint_steps 10000 --keep_checkpoint 10 \
    --train_steps 300000 --param_init 0  --param_init_glorot --max_generator_batches 32 \
    --batch_size 4096 --batch_type tokens --normalization tokens --max_grad_norm 0  --accum_count 4 \
    --optim adam --adam_beta1 0.9 --adam_beta2 0.998 --decay_method noam --warmup_steps 8000  \
    --learning_rate 2 --label_smoothing 0.0 --report_every 1000 --valid_steps 10000 \
    --layers 4 --rnn_size 256 --word_vec_size 256 --encoder_type transformer --decoder_type transformer \
    --dropout 0.1 --position_encoding --share_embeddings \
    --global_attention general --global_attention_function softmax --self_attn_type scaled-dot \
    --heads 8 --transformer_ff 2048 --master_port 10001
