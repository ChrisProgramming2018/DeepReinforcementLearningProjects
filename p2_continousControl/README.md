[//]: # (Image References)

[image1]: https://user-images.githubusercontent.com/10624937/43851024-320ba930-9aff-11e8-8493-ee547c6af349.gif "Trained Agent"
[image2]: https://user-images.githubusercontent.com/10624937/43851646-d899bf20-9b00-11e8-858c-29b5c2c94ccc.png "Crawler"


# Project 2: Continuous Control

### Introduction

For this project, you will work with the [Reacher](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Learning-Environment-Examples.md#reacher) environment.

![Trained Agent][image1]

In this environment, a double-jointed arm can move to target locations. A reward of +0.1 is provided for each step that the agent's hand is in the goal location. Thus, the goal of your agent is to maintain its position at the target location for as many time steps as possible.

The observation space consists of 33 variables corresponding to position, rotation, velocity, and angular velocities of the arm. Each action is a vector with four numbers, corresponding to torque applicable to two joints. Every entry in the action vector should be a number between -1 and 1.

### Distributed Training

For this project, we will provide you with two separate versions of the Unity environment:

- The second version contains 20 identical agents, each with its own copy of the environment.  


#### Option 1: Solve the First Version

The task is episodic, and in order to solve the environment,  your agent must get an average score of +30 over 100 consecutive episodes.

### Instructions
1. Clone the repository and navigate to the downloaded folder.
```	
     git clone https://github.com/ChrisProgramming2018/DeepReinforcementLearningProjects.git
     cd DeepReinforcementLearningProjects
```	
2. Create a virtual enviroment 
```	
     virtualenv -p python3 p2_continousControl
```
3. Activate the Enviroment
```
	source p2_continousControl/bin/activate
	cd p2_continousControl
```

4.  **Install all dependencies**, use the requirements.txt file

```
	pip3 install -r requirements.txt
```
5.  Download the environment for linux 


- **_Version 2: Twenty (20) Agents_**
        - Linux: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher_Linux.zip)

```
     download it to the same folder p2_continousControl
    
     unzip  Reacher_Linux.zip
     
     rm Reacher_Linux.zip
```
6.  Train agent 
```
   python3 ddpg_main.py
```
7.  Watch all 20 smart agent moving the arms 
 ```
   python3 watch_samrt_agent.py 
``` 
