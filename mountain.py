import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt

#this code is adapted from https://www.kaggle.com/code/abedi756/car-mountain-rainforcement

env = gym.make('MountainCar-v0')

# env.observation_space = Box([-1.2  -0.07], [0.6  0.07], (2,), float32)
# We will discretize the state space into .1 increments for position (first coordinate) and .01 increments for velocity (second coordinate)
# this gives a state space consisting of 19*15 = 285 states

def discretize(state):
    temp = (state - env.observation_space.low) * np.array([10,100])
    return np.round(temp, 0).astype(int)

# we will train an expert using Q learning
# The goal_reward input sets the reward for reaching the goal state, which is not included by default in the MountainCar implementation
# The MountainCar implementation in gym gives a -1 reward per time step to penalize taking longer to reach the goal, so we expect rewards to be negative in general

def QLearning(env, alpha, discount, epsilon, goal_reward, episodes):
    # initialize Q function with random values to encourage initial exploration
    Q = np.random.uniform(low = -.5, high = .5, size = (19, 15, 3))
    #Q = np.load("Q_mat1.npy")
    rewards = []
    for i in range(episodes):
        # reset for new training episode
        found_goal = False
        total_reward = 0
        curr = discretize(env.reset()[0])
        #track progress if desired
        if i%100==0:
            print(f"{100*i/episodes}% done")

        #train for one episode
        while not found_goal:   
            if np.random.random() < epsilon and i < episodes//2:
                action = np.random.randint(0, env.action_space.n)
            else:
                action = np.argmax(Q[curr[0], curr[1]])
                
            new_state, reward, found_goal, done, _ = env.step(action) 
            new_state = discretize(new_state)
            if found_goal:
                reward = goal_reward
                
            update = reward + discount*np.max(Q[new_state[0], new_state[1]])
            Q[curr[0], curr[1], action] = (1-alpha)*Q[curr[0], curr[1], action] + alpha*update
            total_reward += reward
            curr = new_state

        #decrement exploration parameter 
        if epsilon > .01:
            epsilon -= epsilon/episodes
        rewards.append(total_reward)
    return rewards, Q

# extract policy from learned Q function
def extract_policy(Q):
    policy = {}
    for i in range(19):
        for j in range(15):
            policy[(i,j)] = np.argmax(Q[i][j])
    return policy

# sample trajectories using expert policy
def extract_trajectory(n):
    trajectories = []
    for _ in range(n):
        trajectory = []
        curr = tuple(discretize(env.reset()[0]))
        found_goal = False
        while not found_goal:
            action = policy[curr]
            trajectory.append((curr,action))
            new_state, _, found_goal, _, _ = env.step(action) 
            curr = tuple(discretize(new_state))
        trajectories.append(trajectory)
    return trajectories

def test(n):
    trials = []
    for _ in range(n):
        counter = 0
        curr = discretize(env.reset()[0])
        found_goal = False
        while not found_goal:
            action = policy[tuple(curr)]
            new_state, _, found_goal, _, _ = env.step(action)
            curr = discretize(new_state)
            counter += 1
        trials.append(counter)
    return trials

rewards, Q = QLearning(env, .2, 0.9, 0.01, 100, 10000)
print(np.median(rewards))
np.save("Q_mat4", Q)
#print(Q)
averaged_rewards = [sum(rewards[100*i:100*(i+1)])/100 for i in range(len(rewards)//100)]
#plt.plot(np.arange(len(rewards)), rewards)
plt.plot(100*(np.arange(len(averaged_rewards))+1), averaged_rewards)
plt.xlabel('Episodes')
plt.ylabel('Reward')
#plt.title('Reward vs Episodes')
plt.show()

"""
In order to test our ideas in a continuous environment, we decided to work with the Mountain Car problem. In this problem, a car is initially placed in a 'valley' between two 'mountains' and can drive in either direction. The goal is to drive up the right-side mountain, but the car does not have enough power to drive up initially and must instead build up momentum by driving back and forth.

We used OpenAI's gymnasium package for our MountainCar implementation. This is an MDP with a continuous state space consisting of pairs (x,v) where x refers to the car's position (ranging from -1.2 to .6 in the openAI implementation with any value of .5 considered having reached the goal state) and v refers to the car's velocity (ranging from -.07 to .07 in the openAI implementation). The car may take one of three actions at each step:

0) do not accelerate in either direction
1) accelerate left
2) accelerate right

and is given a reward of -1 per time step taken. This negative reward per time step encourages solutions that reach the goal state more quickly.

To train an expert for this problem, we decided to discretize the state space by considering 19 0.1 increments in the position and 15 .01 increments in the velocity. For example, the state (-1.11, -.0654) would correspond to the discretized encoding (1,0) while the state (.54, .056) would correspond to the discretized encoding (17,13). Using this discretization, we trained an expert with Q learning, which has updates to the value function of the following form:

Q^new(s_t, a_t) = (1-alpha)Q(s_t,a_t) + alpha(reward + discount*max_a Q(s_t+1,a))

s_t = state at timestep t
a_t = action at timestep t
alpha = learning rate
reward = reward for taking action a_t in state s_t
discount = discount factor
s_t+1 = state at timestep t+1 after taking action a_t from state s_t

We tried various parameter settings, but the one that seemed to yield the best results was

learning rate = .2 (how quickly to update the values based on the results of an action)
discount factor = .9 (how much to discount future rewards)
epsilon = .01 (how often to choose a random action, we made this parameter decay)
goal_reward = 100 (what reward to attach to reaching the goal state)
episodes = 10000 (number of trials to run Q learning over)
"""
