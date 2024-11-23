episodes=10000000


python main_DQN.py \
--simulation hard \
--model hard_DQN \
--episodes $episodes

python main_DQN.py \
--simulation hard \
--model hard_DQN \
--test \
--episodes 100


python main_DQN.py \
--simulation hard \
--model hard_DQN \
--test \
--animate \
--title sim=hard_train_eps=$episodes


