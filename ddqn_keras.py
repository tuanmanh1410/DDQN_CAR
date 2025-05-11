from tensorflow.keras.models import Sequential, load_model, clone_model
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
import numpy as np
import tensorflow as tf

class ReplayBuffer(object):
    def __init__(self, max_size, input_shape, n_actions, discrete=False):
        self.mem_size = max_size
        self.mem_cntr = 0
        self.discrete = discrete
        self.state_memory = np.zeros((self.mem_size, input_shape))
        self.new_state_memory = np.zeros((self.mem_size, input_shape))
        dtype = np.int8 if self.discrete else np.float32
        self.action_memory = np.zeros((self.mem_size, n_actions), dtype=dtype)
        self.reward_memory = np.zeros(self.mem_size)
        self.terminal_memory = np.zeros(self.mem_size, dtype=np.float32)

    def store_transition(self, state, action, reward, state_, done):
        index = self.mem_cntr % self.mem_size
        self.state_memory[index] = state
        self.new_state_memory[index] = state_
        # store one hot encoding of actions, if appropriate
        if self.discrete:
            actions = np.zeros(self.action_memory.shape[1])
            actions[action] = 1.0
            self.action_memory[index] = actions
        else:
            self.action_memory[index] = action
        self.reward_memory[index] = reward
        self.terminal_memory[index] = 1 - done
        self.mem_cntr += 1

    def sample_buffer(self, batch_size):
        max_mem = min(self.mem_cntr, self.mem_size)
        batch = np.random.choice(max_mem, batch_size)

        states = self.state_memory[batch]
        actions = self.action_memory[batch]
        rewards = self.reward_memory[batch]
        states_ = self.new_state_memory[batch]
        terminal = self.terminal_memory[batch]

        return states, actions, rewards, states_, terminal

class DDQNAgent:
    def __init__(self,
                 alpha, gamma, n_actions, epsilon, batch_size,
                 input_dims, epsilon_dec=0.9995, epsilon_end=0.01,
                 mem_size=100000, fname='ddqn_model.h5', replace_target=500):
        # core hyperparameters
        self.alpha       = alpha               # learning rate
        self.gamma       = gamma               # discount factor
        self.epsilon     = epsilon             # initial ε for ε-greedy
        self.epsilon_dec = epsilon_dec
        self.epsilon_min = epsilon_end
        self.batch_size  = batch_size
        self.replace_target = replace_target   # how often to sync target net

        # internal trackers
        self.learn_step_counter = 0
        self.model_file         = fname
        self.action_space       = list(range(n_actions))

        # replay buffer (discrete actions → one-hot storage)
        self.memory = ReplayBuffer(mem_size, input_dims, n_actions, discrete=True)

        # build eval and target networks
        self.brain_eval   = Brain(input_dims, n_actions, batch_size, learning_rate=alpha)
        self.brain_target = Brain(input_dims, n_actions, batch_size, learning_rate=alpha)

    def remember(self, state, action, reward, state_, done):
        """Store a transition in replay buffer."""
        self.memory.store_transition(state, action, reward, state_, done)

    def choose_action(self, state):
        """ε-greedy action selection."""
        state = np.array(state)[np.newaxis, :]  # shape (1, input_dims)
        if np.random.random() < self.epsilon:
            return np.random.choice(self.action_space)
        q_vals = self.brain_eval.predict(state)
        return np.argmax(q_vals)

    def learn(self):
        """Sample a batch and do a Double-DQN update."""
        if self.memory.mem_cntr < self.batch_size:
            return

        # 1) sample batch
        states, actions, rewards, states_, dones = \
            self.memory.sample_buffer(self.batch_size)

        # 2) compute indices and Q-values
        action_indices = np.argmax(actions, axis=1)
        q_pred         = self.brain_eval.predict(states)       # Q(s,·)
        q_next         = self.brain_target.predict(states_)    # Q'(s',·)
        q_eval_next    = self.brain_eval.predict(states_)      # Q(s',·)

        # 3) build target Q-value array
        max_actions = np.argmax(q_eval_next, axis=1)
        q_target    = q_pred.copy()
        batch_idx   = np.arange(self.batch_size, dtype=np.int32)

        # TD update (done==0 for terminal, 1 for non-terminal)
        q_target[batch_idx, action_indices] = (
            rewards
            + self.gamma * q_next[batch_idx, max_actions] * dones
        )

        # 4) train eval network
        _ = self.brain_eval.train(states, q_target)

        # 5) ε‐decay
        self.epsilon = (
            self.epsilon * self.epsilon_dec
            if self.epsilon > self.epsilon_min
            else self.epsilon_min
        )

        # 6) target network sync
        self.learn_step_counter += 1
        if self.learn_step_counter % self.replace_target == 0:
            self.update_network_parameters()

    def update_network_parameters(self):
        """Copy weights from eval → target and reset counter."""
        self.brain_target.model.set_weights(
            self.brain_eval.model.get_weights()
        )
        self.learn_step_counter = 0

    def save_model(self):
        """Persist only the evaluation network to disk."""
        self.brain_eval.model.save(self.model_file)

    def load_model(self, for_training: bool = False):
        """
        Load eval network (no compile for pure inference).
        If for_training=True, re-compile with MSE + Adam(alpha).
        Then clone weights into the target network.
        """
        # 1) load eval network without optimizer state
        self.brain_eval.model = load_model(self.model_file, compile=False)

        # 2) optionally re-compile for further training
        if for_training:
            self.brain_eval.model.compile(
                loss='mse',
                optimizer=Adam(learning_rate=self.alpha)
            )

        # 3) clone to target network
        self.brain_target.model = tf.keras.models.clone_model(
            self.brain_eval.model
        )
        self.brain_target.model.set_weights(
            self.brain_eval.model.get_weights()
        )

        # 4) for pure inference, force greedy
        if not for_training:
            self.epsilon = 0.0

        # 5) reset sync counter
        self.learn_step_counter = 0

class Brain:
    def __init__(self, n_states, n_actions, batch_size, learning_rate):
        self.n_states      = n_states
        self.n_actions     = n_actions
        self.batch_size    = batch_size
        self.learning_rate = learning_rate
        self.model         = self._build_model()

    def _build_model(self):
        model = Sequential()
        # specify input_shape on the very first layer:
        model.add(Dense(256, activation='relu', input_shape=(self.n_states,)))
        # optional second layer:
        model.add(Dense(256, activation='relu'))
        # linear outputs for Q-values:
        model.add(Dense(self.n_actions, activation=None))

        optimizer = Adam(learning_rate=self.learning_rate)
        model.compile(loss='mse', optimizer=optimizer)
        return model

    def train(self, x, y, epoch=1, verbose=0):
        self.model.fit(x, y,
                       batch_size=self.batch_size,
                       epochs=epoch,
                       verbose=verbose)

    def predict(self, s):
        return self.model.predict(s)

    def copy_weights(self, source_brain):
        """Copy weights from another Brain instance."""
        self.model.set_weights(source_brain.model.get_weights())