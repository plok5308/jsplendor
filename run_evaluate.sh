# play 100 games
python evaluate_agents.py \
    --player_model_path ./pretrained/hard_11_no_reserve.zip \
    --opponent_model_path ./pretrained/medium_reserve_once.zip \
    --reserve_masking player \
    --n_episodes 100 \

# # play single game with logs
# python evaluate_agents.py \
#     --player_model_path ./pretrained/hard_11_no_reserve.zip \
#     --opponent_model_path ./pretrained/medium_reserve_once.zip \
#     --reserve_masking player \
#     --n_episodes 1 \
#     --verbose \

