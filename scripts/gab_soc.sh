#!/bin/bash

# training with regularzing SOC explanations

if [[ -n "$1" ]]; then
    current_seed=$1
else
    current_seed=0
fi


if [[ -n "$2" ]]; then
    max_seeds=$2
else
    max_seeds=4
fi


reg_strength=0.1

echo "reg_strength is ${reg_strength}"
current_seed=0
while(( $current_seed < $max_seeds ))
do
    python run_model.py --do_train --do_lower_case --data_dir ./data/project_data/gab/0.1 --bert_model bert-base-uncased --max_seq_length 128 --train_batch_size 32 --learning_rate 2e-5 --num_train_epochs 15 --early_stop 5 --output_dir runs/gab_0.1_seed_${current_seed} --seed ${current_seed} --task_name gab --hiex_add_itself --reg_explanations --nb_range 5 --sample_n 5 --negative_weight 0.1 --reg_strength ${reg_strength}
    let current_seed++
done
