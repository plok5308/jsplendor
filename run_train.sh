
# first phase (train model with both masking)
python train_self_play.py --reserve_masking both

# second phase (train model with opponent masking)
#python train_self_play.py --reserve_masking opponent --load_model {model_path}
