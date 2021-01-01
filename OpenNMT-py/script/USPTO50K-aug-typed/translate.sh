dataset=USPTO50K-aug-typed
suffix=test-prediction
python ../../onmt/bin/translate.py --model ../../experiments/checkpoints/${dataset}/average_model.pt \
    --gpu 0  --src ../../data/${dataset}/src-${suffix}.txt \
    --output ../../experiments/results/predictions_on_${dataset}_${suffix}.txt \
    --beam_size 50  --n_best 50 \
    --batch_size 8 --replace_unk --max_length 300