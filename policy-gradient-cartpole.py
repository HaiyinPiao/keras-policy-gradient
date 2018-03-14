import gym
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Reshape, Flatten
from keras.optimizers import Adam
from keras.layers.convolutional import Convolution2D

class PGAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.gamma = 0.99
        self.learning_rate = 0.001
        self.states = []
        self.labels = []
        self.rewards = []
        self.probs = []
        self.model = self._build_model()
        self.model.summary()

    def _build_model(self):
        model = Sequential()
        model.add(Dense(24, input_dim=self.state_size, activation='relu', init='he_uniform'))
        model.add(Dense(24, activation='relu', init='he_uniform'))
        model.add(Dense(self.action_size, activation='softmax'))
        model.compile(loss='categorical_crossentropy',
                      optimizer=Adam(lr=self.learning_rate))
        return model

    def remember(self, state, action, prob, reward):
        y = np.zeros([self.action_size])
        y[action] = 1
        self.labels.append(np.array(y).astype('float32'))
        self.states.append(state)
        self.rewards.append(0.0)

    def act(self, state):
        state = state.reshape([1, state.shape[0]])
        aprob = self.model.predict(state, batch_size=1).flatten()
        self.probs.append(aprob)
        action = np.random.choice(self.action_size, 1, p=aprob)[0]
        return action, aprob

    def discount_rewards(self, rewards):
        discounted_rewards = np.zeros_like(rewards)
        running_add = 0
        for t in reversed(range(0, rewards.size)):
            if rewards[t] != 0:
                running_add = 0
            running_add = running_add * self.gamma + rewards[t]
            discounted_rewards[t] = running_add
        return discounted_rewards

    def train(self):
        labels = np.vstack(self.labels)
        rewards = np.vstack(self.rewards)
        rewards = self.discount_rewards(rewards)
        rewards = rewards / np.std(rewards)
        labels *= -rewards
        X = np.squeeze(np.vstack([self.states]))
        Y = np.squeeze(np.vstack([labels]))
        self.model.train_on_batch(X, Y)
        self.states, self.probs, self.labels, self.rewards = [], [], [], []

    def load(self, name):
        self.model.load_weights(name)

    def save(self, name):
        self.model.save_weights(name)

if __name__ == "__main__":
    env = gym.make("CartPole-v1")
    state = env.reset()
    score = 0
    episode = 0

    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n
    agent = PGAgent(state_size, action_size)
    agent.load('cartpole_spg_haiyin.h5')
    while True:
        env.render()

        action, prob = agent.act(state)
        next_state, reward, done, info = env.step(action)
        score += reward
        agent.remember(state, action, prob, reward)
        state = next_state

        if done:
            episode += 1
            agent.rewards[-1] = score
            agent.train()
            print('Episode: %d - Score: %f.' % (episode, score))
            score = 0
            state = env.reset()
            if episode > 1 and episode % 50 == 0:
                agent.save('cartpole_spg_haiyin.h5')
