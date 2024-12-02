## Spaceship-Navigation

# Double Deep Q-Learning & Proximal Policy Optimization
* see the PPO branch for the PPO code and animations

We use a deep Q-netowrk to train a spaceship agent to navigate a dynamic star system. We present an exploration of the challenges and ideas that we encountered throughout our investigation of the problem.

<!-- <img src="https://user-images.githubusercontent.com/63081584/235216113-a238a10a-cf90-4307-ac82-93948e089d7f.gif" width="500" height="500"/> -->

<p float="left">
  <img src="animations/1_model=DoubleDQN_sim=mass_easy_eps=9000_pos_r_0_train_eps=9000_DoubleDQN copy.gif" width="300" height="300"/>
  <img src="https://user-images.githubusercontent.com/63081584/235218435-636983e3-5a1f-4d66-8bf6-a5f31f8aee8e.gif" width="300" height="300"/> 
</p>


## Quick Start

python main.py --model baseline --simulation sim1 --kernel_size 3 --test

![7_hmm](https://user-images.githubusercontent.com/63081584/235223870-c7e3720c-8a73-4f5a-8205-fd7ee0f57125.gif)


<!-- <img src="https://user-images.githubusercontent.com/63081584/235217950-e5574d0d-3622-49d9-a8cc-4a57516107ba.gif" width="500" height="500"/>  -->

---

Based on code by Soroco, Mauricio and Shpilevskiy, Frederick and Vandervalk, Joel: [A SpaceQ Rocket explores Deep Space with Deep Q-Networks](https://github.com/msoroco/c440-project). 2024
