## Semantic Predictive Learning on Autonomous Driving

An implementation of flexible semantic prediction learning
    
    1. Option to choose either predict segmentation or predict latent representation
    2. Option to choose number of steps to predict, with 0 step corresponding to imitation learning
    3. Option to do discrete or contiuous control
    4. Option to add DQN as a bound for action selection
    5. Option to predict x,y,z relative movement for flexible control
    6. Option to choose whether to use distance to center of the road as a metric for traveling distance
    7. Option to train and test separately, so that exploration won't affect performance
