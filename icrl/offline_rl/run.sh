for i in {1..10}; do
    python /home/guorui/jax-rl/icrl/offline_rl/discrete_dcrl_new_batch.py --seed $i &
done
wait 