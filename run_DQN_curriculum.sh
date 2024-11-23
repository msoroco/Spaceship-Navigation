sim=(
    hard1
    hard2
    hard3
)
episodes=(
    5000
    5000
    10000
)
experiment_name=curriculum_hard_DQN
model=$experiment_name
wandb_project=spaceship

touch runid_$experiment_name.txt
truncate -s 0 runid_$experiment_name.txt

for i in "${!sim[@]}"; do
    s=${sim[$i]}
    e=${episodes[$i]}
    
    if [ $i -eq 0 ]; then
        start_episode=0
    else
        start_episode=${episodes[$((i-1))]}
    fi

    python main_DQN.py \
    --simulation $s \
    --model $model \
    --episodes $e \
    --start_episode $start_episode \
    --experiment_name $experiment_name \
    --wandb_project $wandb_project

    python main_DQN.py \
    --simulation $s \
    --model $model \
    --test \
    --episodes 1 \
    --experiment_name $experiment_name \
    --wandb_project $wandb_project

    python main_DQN.py \
    --simulation $s \
    --model $model \
    --test \
    --animate \
    --title sim=${s}_train_eps=$e
done








