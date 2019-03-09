"""
Script for the data structure for saveing the
agent experince in a prioritized replay buffer
"""
from collections import namedtuple
import torch
import numpy as np

Transition = namedtuple('Transition', ('timestep', 'state', 'action', 'reward', 'nonterminal'))
BLANK_TRANS = Transition(0, torch.zeros(37, dtype=torch.float32), None, 0, False)


class SegmentTree(object):
    """ Class for Segment tree data structure where parent node values are sum
    of children node values
    """
    def __init__(self, size):
        """ Set all self variables
        Args:
            param1 (int): ammount of leaf nodes
        """
        self.index = 0
        self.size = size
        self.full = False  # Used to track actual capacity
        self.sum_tree = np.zeros((2 * size - 1, ), dtype=np.float32)  # Initialise fixed size tree with all (priority) zeros
        self.data = np.array([None] * size)  # Wrap-around cyclic buffer
        self.max = 1  # Initial max value to return (1 = 1^omega)

    def _propagate(self, index, value):
        """ propagetes the value up the tree given the index

         Args:
             param1 (int): index
             param2 (float): value

        """
        parent = (index - 1) // 2
        left, right = 2 * parent + 1, 2 * parent + 2
        self.sum_tree[parent] = self.sum_tree[left] + self.sum_tree[right]
        if parent != 0:
            self._propagate(parent, value)

    def update(self, index, value):
        """ insert a new value to the given
        Args:
            param1 (int): index
            param2 (float): value

        """
        self.sum_tree[index] = value  # Set new value
        self._propagate(index, value)  # Propagate value
        self.max = max(value, self.max)

    def append(self, data, value):
        """ insert a new value to the given
        Args:
            param1 (tuple): data (torch tensor, int, int, bool)
            param2 (float): value

        """
        self.data[self.index] = data  # Store data in underlying data structure
        self.update(self.index + self.size - 1, value)  # Update tree
        self.index = (self.index + 1) % self.size  # Update index
        self.full = self.full or self.index == 0  # Save when capacity reached
        self.max = max(value, self.max)

    def _retrieve(self, index, value):
        """ seaches for the location of the value in the tree
            returns position
        Args:
            param1 (int): index
            param2 (float): value
        Returns:
            int  :  position in the tree
        """
        left, right = 2 * index + 1, 2 * index + 2
        if left >= len(self.sum_tree):
            return index
        elif value <= self.sum_tree[left]:
            return self._retrieve(left, value)
        else:
            return self._retrieve(right, value - self.sum_tree[left])


    def find(self, value):
        """ seaches for a value in the sum tree
            returns value data index and tree index
        Args:
            param1 (int): value
        Returns:
            tuple  :  value, data index, tree index
        """
        index = self._retrieve(0, value)  # Search for index of item from root
        data_index = index - self.size + 1
        return (self.sum_tree[index], data_index, index)  # Return value, data index, tree index


    def get(self, data_index):
        """ returns data for a given index
        Args:
            param1 (int): data_index
        Returns:
            data  :  value
        """
        return self.data[data_index % self.size]


    def total(self):
        """ returns the root node the sum of the values

        Returns:
            int  :  sum of the tree values
        """
        return self.sum_tree[0]
# ______________________________________________________________________________________________
class ReplayMemory(object):
    """ Class to save experience  """
    def __init__(self, args, capacity):
        """ init all member variables
        Args:
            param1 (parser):  values from command line
            param2 (int):  maximum buffer
        """
        self.device = args.device
        self.capacity = capacity
        self.history = args.history_length
        self.discount = args.discount
        self.n_step = args.multi_step
        self.priority_weight = args.priority_weight  # Init impor.  beta, to 1 over
        self.priority_exponent = args.priority_exponent
        self.t_c = 0  # Internal episode timestep counter
        self.transitions = SegmentTree(capacity)  # Store transitions  within a sum tree


    def append(self, state, action, reward, terminal):
        """ add new experience from agent
        Args:
            param1 (torch.tensor):  state
            param2 (int):  action
            param3 (int):  reward
            param4 (bool): terminal state
        """
        state = state.float()
        self.transitions.append(Transition(self.t_c, state, action, reward, not terminal), self.transitions.max)
        self.t_c = 0 if terminal else self.t_c + 1  # Start new episodes with t = 0

    def _get_transition(self, idx):
        """ Returns a transition with blank states where appropriate
        Args:
            param1 (int):  idx

        Returns:
            transition : tuple(state, action, reward, done)
        """
        transition = np.array([None] * (self.history + self.n_step))
        transition[self.history - 1] = self.transitions.get(idx)
        for i in range(self.history - 2, -1, -1):  # e.g. 2 1 0
            if transition[i + 1].timestep == 0:
                transition[i] = BLANK_TRANS  # If future frame has timestep 0
            else:
                transition[i] = self.transitions.get(idx - self.history + 1 + i)
        for i in range(self.history, self.history + self.n_step):  # e.g. 4 5 6
            if transition[i - 1].nonterminal:
                transition[i] = self.transitions.get(idx - self.history + 1 + i)
            else:
                transition[i] = BLANK_TRANS  # If prev (next) frame is terminal
        return transition

    def _get_sample_from_segment(self, segment, i):
        """ Returns a valid sample from the segment
        Args:
            param1 (int):  segment
            param1 (int):  i

        Returns:
             prob, idx, tree_idx, state, action, reward, next_state, non trminal

        """
        valid = False
        while not valid:
            sample = np.random.uniform(i * segment, (i + 1) * segment)  # Uniformly in a segment
            prob, idx, tree_idx = self.transitions.find(sample)  # sample from tree with un-norma
            if (self.transitions.index - idx) % self.capacity > self.n_step and (idx - self.transitions.index) % self.capacity >= self.history and prob != 0:
                valid = True  #  conditions are valid but extra cons around buffer index 0
        # Retrieve all required transition data (from t - h to t + n)
        transition = self._get_transition(idx)
        # Create un-discretised state and nth next state
        state = torch.stack([trans.state for trans in transition[:self.history]]).to(dtype=torch.float32, device=self.device)
        next_state = torch.stack([trans.state for trans in transition[self.n_step:self.n_step + self.history]]).to(dtype=torch.float32, device=self.device)
        # Discrete action to be used as index
        action = torch.tensor([transition[self.history - 1].action], dtype=torch.int64, device=self.device)
        rew = torch.tensor([sum(self.discount ** n * transition[self.history + n- 1].reward for n in range(self.n_step))], dtype=torch.float32, device=self.device)
        # Mask for non-terminal nth next states
        nonterminal = torch.tensor([transition[self.history + self.n_step - 1].nonterminal], dtype=torch.float32, device=self.device)
        return prob, idx, tree_idx, state, action, rew, next_state, nonterminal


    def sample(self, batch_size):
        """ Returns experince samples in the amount of the batch_size
        Args:
            param1 (int):  batch_size

        Returns:
             tree_idxs, states, actions, returns, next_state, non trminal, weights
        """
        p_total = self.transitions.total()  # Retrieve sum of all priorities (used to create a normalised probability distribution)
        segment = p_total / batch_size  # Batch size number of segments, based on sum over all probabilities
        batch = [self._get_sample_from_segment(segment, i) for i in range(batch_size)]  # Get batch of valid samples
        probs, _, tree_idxs, states, actions, returns, next_states, nonterminals = zip(*batch)
        states, next_states, = torch.stack(states), torch.stack(next_states)
        actions, returns, nonterminals = torch.cat(actions), torch.cat(returns), torch.stack(nonterminals)
        probs = np.array(probs, dtype=np.float32) / p_total  # Calculate normalised probabilities
        capacity = self.capacity if self.transitions.full else self.transitions.index
        weights = (capacity * probs) ** -self.priority_weight  # Compute importance-sampling weights w
        weights = torch.tensor(weights / weights.max(), dtype=torch.float32, device=self.device)  # Normalise by max importance-sampling weight from batch
        return tree_idxs, states, actions, returns, next_states, nonterminals, weights



    def update_priorities(self, idxs, priorities):
        """"  changes the priorities in the tree
        Args:
            param1(list): idxs
            param2(list): priorities

        """
        priorities = np.power(priorities, self.priority_exponent)
        [self.transitions.update(idx, priority) for idx, priority in zip(idxs, priorities)]

    def __iter__(self):
        """set up internal state for iterator"""
        self.current_idx = 0
        return self

    def __next__(self):
        """ Returns valid states for validation

        Return:
              states
        """
        if self.current_idx == self.capacity:
            raise StopIteration
        # Create stack of states
        state_stack = [None] * self.history
        state_stack[-1] = self.transitions.data[self.current_idx].state
        prev_timestep = self.transitions.data[self.current_idx].timestep
        for i in reversed(range(self.history - 1)):
            if prev_timestep == 0:
                state_stack[i] = BLANK_TRANS.state  # If future frame has timestep 0
            else:
                state_stack[i] = self.transitions.data[self.current_idx + i - self.history + 1].state
            prev_timestep -= 1
        state = torch.stack(state_stack, 0).to(dtype=torch.float32, device=self.device).div_(255)  # Agent will turn into batch
        self.current_idx += 1
        return state
