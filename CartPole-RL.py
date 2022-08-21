import gym
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.optimizers import Adam
from rl.agents import DQNAgent
from rl.policy import BoltzmannQPolicy
from rl.memory import SequentialMemory


def build_model(states, actions):
    """build model with 2 hidden layers"""
    model = Sequential()
    model.add(Flatten(input_shape=states))
    model.add(Dense(24, activation="relu"))
    model.add(Dense(24, activation="relu"))
    model.add(Dense(actions, activation="linear"))
    return model


def build_agent(model, actions):
    """learn from the environment"""
    policy = BoltzmannQPolicy()
    memory = SequentialMemory(limit=50000, window_length=1)
    dqn = DQNAgent(
        model=model,
        memory=memory,
        policy=policy,
        nb_actions=actions,
        nb_steps_warmup=100,
        target_model_update=1e-2,
    )
    return dqn


def main():
    # Create env
    env = gym.make("CartPole-v0")

    # Get states and actions
    states = list(env.observation_space.shape)
    states.insert(0, 1)
    actions = env.action_space.n

    # Build model
    model = build_model(states, actions)
    # model.summary()

    # Build agent
    dqn = build_agent(model, actions)
    dqn.compile(Adam(lr=1e-3), metrics=["mae"])

    # Train agent
    steps = 10000
    dqn.fit(env, nb_steps=steps, visualize=False, verbose=1)
    dqn.save_weights("dqn_weights.h5f", overwrite=True)

    # Load weights
    # dqn.load_weights("dqn_weights.h5f")

    # scores = dqn.test(env, nb_episodes=100, visualize=False)
    # print(np.mean(scores.history["episode_reward"]))
    _ = dqn.test(env, nb_episodes=10, visualize=True)

    # Close environment properly
    env.close()


if __name__ == "__main__":
    main()
