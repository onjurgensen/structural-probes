#!/bin/bash

yaml_path="gpt2_configs"
output_path="gpt2_results/mean-pooling"

cd /home/fleetprobe/structural-probes

# skip=12
# count=0

for yaml in $yaml_path/*.yaml; do

    # if [ $count -lt $skip ]; then
    #     count=$((count + 1))
    #     continue
    # fi

    echo "Updating only embeddings paths and reporting root in $yaml"
    awk -v newroot="./$output_path/" '
    BEGIN {in_embeddings=0; in_reporting=0}
    /^  embeddings:/ {in_embeddings=1; in_reporting=0; print; next}
    /^reporting:/ {in_reporting=1; in_embeddings=0; print; next}
    in_embeddings && /^[ ]+train_path:/ {$0="    train_path: gpt2_train_embeddings_mean.hdf5"}
    in_embeddings && /^[ ]+dev_path:/ {$0="    dev_path: gpt2_dev_embeddings_mean.hdf5"}
    in_embeddings && /^[ ]+test_path:/ {$0="    test_path: gpt2_test_embeddings_mean.hdf5"}
    in_embeddings && NF==0 {in_embeddings=0}
    in_reporting && /^[ ]+root:/ {$0="  root: " newroot}
    in_reporting && NF==0 {in_reporting=0}
    {print}
    ' "$yaml" > tmp.yaml && mv tmp.yaml "$yaml"
    echo "Running experiment with $yaml"
    python structural-probes/run_experiment.py "$yaml"
done

