dataset=USPTO50K-aug-typed
suffix=test-prediction
python  ../../score_predictions.py  --beam_size 50 --invalid_smiles \
    --predictions  ../../experiments/results/predictions_on_${dataset}_${suffix}.txt \
    --targets  ../../data/${dataset}/tgt-test.txt \
    --sources  ../../data/${dataset}/src-${suffix}.txt
