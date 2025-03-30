python evaluate_agents.py \
    --model1_type linear \
    --model1_path ./logs/allow_reserve_opponent/best_models/model_gen_1_last.zip \
    --model2_type linear \
    --model2_path ./logs/allow_reserve_opponent/best_models/model_gen_1_last.zip \
    --reserve_masking player \
    --n_episodes 1 \
    --verbose \