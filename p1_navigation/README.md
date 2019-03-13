[//]: # (Image References)

[image1]: https://user-images.githubusercontent.com/10624937/42135619-d90f2f28-7d12-11e8-8823-82b970a54d7e.gif "Trained Agent"

# Project 1: Navigation

### Introduction

In this project, I used Deep-Reinforcement Learning to train an agent to navigate (and collect bananas!) in a large, square world.  

![Trained Agent][image1]

A reward of +1 is provided for collecting a yellow banana, and a reward of -1 is provided for collecting a blue banana.  Thus, the goal of your agent is to collect as many yellow bananas as possible while avoiding blue bananas.  

The state space has 37 dimensions and contains the agent's velocity, along with ray-based perception of objects around agent's forward direction.  Given this information, the agent has to learn how to best select actions.  Four discrete actions are available, corresponding to:
- **`0`** - move forward.
- **`1`** - move backward.
- **`2`** - turn left.
- **`3`** - turn right.
### The Goal 
The task is episodic, and in order to solve the environment, your agent must get an average score of +13 over 100 consecutive episodes.

### Getting Started

1. Download the environment from one of the links below.  You need only select the environment that matches your operating system:
    - Linux: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Linux.zip)
    - Mac OSX: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana.app.zip)
    - Windows (32-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Windows_x86.zip)
    - Windows (64-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Windows_x86_64.zip)
    
    (_For Windows users_) Check out [this link](https://support.microsoft.com/en-us/help/827218/how-to-determine-whether-a-computer-is-running-a-32-bit-version-or-64) if you need help with determining if your computer is running a 32-bit version or 64-bit version of the Windows operating system.



### Instructions
1. Clone the repository and navigate to the downloaded folder.
```	
     git clone https://github.com/ChrisProgramming2018/DeepReinforcementLearningProjects.git
     cd DeepReinforcementLearningProjects
```	
2. Create a virtual enviroment 
```	
     virtualenv -p python3 p1_navigation
```
3. Activate the Enviroment
```
	source p1_navigation/bin/activate
```

4.  **Install all dependencies**, use the requirements.txt file

```
	pip3 install -r requirements.txt
```
5.  Download the environment for linux 
```
	click on the link
```
```
     download it to the same folder p1_navigation
     unzip Banana_Linux.zip 
     rm Banana_Linux.zip 
```
6.  Use  different  hyper parameter values to find the best 
```
	python3  hyperparametersearch.py
```
7. Train agent with the "best" hyper parameter   
```
	python3  training.py
```
8. Watch smart agent (load trained network weights) in the environment collecting the yellows bananas   
```
	python3 smart_agent.py 
```
9. Compare the 3 different agents
```
	python3 compare.py
```

10. Issues
    In case of any problem check if the correct parameters are set and the pathnames are correct
The research paper from the used algorithms

- [x] DQN [[2]](#references)
- [x] Double DQN [[3]](#references)
- [x] Prioritised Experience Replay [[4]](#references)
- [x] Dueling Network Architecture [[5]](#references)
	
	
	
	
	References
----------

[1] [Rainbow: Combining Improvements in Deep Reinforcement Learning](https://arxiv.org/abs/1710.02298)  
[2] [Playing Atari with Deep Reinforcement Learning](http://arxiv.org/abs/1312.5602)  
[3] [Deep Reinforcement Learning with Double Q-learning](http://arxiv.org/abs/1509.06461)  
[4] [Prioritized Experience Replay](http://arxiv.org/abs/1511.05952)  
[5] [Dueling Network Architectures for Deep Reinforcement Learning](http://arxiv.org/abs/1511.06581)  
[6] [Reinforcement Learning: An Introduction](http://www.incompleteideas.net/sutton/book/ebook/the-book.html)  
[7] [A Distributional Perspective on Reinforcement Learning](https://arxiv.org/abs/1707.06887)  
[8] [Noisy Networks for Exploration](https://arxiv.org/abs/1706.10295)  
