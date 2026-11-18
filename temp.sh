for model in raw_bce_dice augmentation_photometric; do
    for seed in 222 333; do
        bash experiments_v4.sh \
            "experiments_output/runs/${model}_seed${seed}"
    done
done
