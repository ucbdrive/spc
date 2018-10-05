import numpy as np
import pdb
import copy

def get_act_samps(num_time, num_actions=6, prev_act=0, num_samples=1000):
    acts = []
    probs = 0
    num = 0
    cnt = 0
    while num < num_samples:
        act, prob = get_act_seq(num_time, num_actions, prev_act)
        act = list(act)
        if act not in acts:
            acts.append(act)
            probs += prob
            cnt = 0
            num += 1
        else:
            cnt +=1
        if cnt >= 100000:
            break
    res = []
    for i in range(len(acts)):
        this_act = acts[i]
        this_act2 = np.zeros((num_time, num_actions))
        for j in range(num_time):
            this_act2[j,this_act[j]] = 1
        res.append(this_act2)
    return np.stack(res), probs

def get_act_seq(num_time, num_actions=6, prev_act=0):
    actions = []
    curr_act = copy.deepcopy(prev_act)
    probs = 1.0
    for i in range(num_time):
        if curr_act in [1,4]:
            p = np.ones(num_actions)/(num_actions*1.0)
        elif curr_act in [0,2,3,5]:
            p = np.ones(num_actions)/2.0
            p[2] = 0
            p[5] = 0
            p[0] = 0
            p[3] = 0
        curr_act = np.random.choice(6, p=list(p/p.sum()))
        actions.append(curr_act)
        if prev_act in [1,4]:
            probs *= 1/(num_actions*1.0)
        elif prev_act in [0,2,3,5]:
            probs *= 1/2.0 
        prev_act = copy.deepcopy(curr_act)
    return np.array(actions), probs
            
def get_action_sample(num_time, num_step_per_time, num_actions=9):
    # num time: number of prediction time
    # num step per time: how many steps are of the same action
    num_step = int(num_time/num_step_per_time)
    args1 = []
    for i in range(num_step):
        args1.append(np.arange(0,num_actions,1))
    args1 = tuple(args1)
    outs1 = np.meshgrid(*args1)
    args2 = []
    for i in range(num_step):
        args2.append(outs1[i].ravel())
    args2 = tuple(args2)
    points = np.c_[args2]
    res = np.repeat(points, [num_step_per_time], axis=1)
    res = res[:,:num_time]
    res2 = np.zeros((res.shape[0], res.shape[1], num_actions))
    res2 = res2.reshape((-1, num_actions))
    res3 = res.reshape((-1))
    res2[range(res2.shape[0]), res3] = 1
    res2 = res2.reshape((res.shape[0], res.shape[1], num_actions))
    return res2  # [9**num_step, num_time]

def get_action_from_probs(batch_size, num_time, prev_action, num_actions=6):
    '''
    action meanings: 0: turn left, accelerate  1: accelerate 2: turn right, accelerate
                    3: turn left               4: do nothing 5: turn right
                    6: turn left, decelerate   7: decelerate 8: turn right, decelerate
    '''
    all_acts = []
    for i in range(batch_size):
        actions = []
        current_act = copy.copy(prev_action)
        for i in range(num_time):
            if current_act in [1,4]:
                p = np.ones(num_actions)*(1-0.7)/(num_actions-2)
                p[1] = 0.35
                p[4] = 0.35
            elif current_act in [0,2]:
                p = np.ones(num_actions)*(1-0.6)/(num_actions-2)
                p[0] = 0.3
                p[2] = 0.3
            elif current_act in [3,5]:
                p = np.ones(num_actions)*(1-0.6)/(num_actions-2)
                p[3] = 0.3
                p[5] = 0.3
            else:
                p = np.ones(num_actions)*(1-0.9)/(num_actions-1)
                p[current_act] = 0.9
            current_act = np.random.choice(6,p=list(p/p.sum()))
            actions.append(current_act)
        all_acts.append(np.array(actions))
    all_acts = np.concatenate(all_acts)
    res2 = np.zeros((batch_size*num_time, num_actions))
    res2[range(res2.shape[0]), all_acts.reshape((-1))] = 1
    res2 = res2.reshape((batch_size, num_time, num_actions))
    return res2

def get_act_prob(action, prob, last_action=None, num_actions=9):
    # action : 1 * num_step 
    num_step = len(action)
    this_prob = 1
    flag = action[:-1]==action[1:]
    flag = np.array(flag)*1
    flag2 = np.minimum(flag+1/5.0, 1.0)
    flag = flag*prob+(1-prob)*2
    flag = np.minimum(flag, 1.0) - (1-prob)
    flag = flag*flag2
    res = np.prod(flag)
    if last_action is None:
        res *= 1/6.0
    else:
        if action[0] == last_action:
            res *= prob
        else:
            res *= (1-prob)/5.0
    return res
