# Adapted from:
# https://github.com/deepmind/acme/blob/master/examples/quickstart.ipynb

from acme import environment_loop
from acme import specs
from acme import wrappers
from acme.agents.tf import r2d2
from acme.agents.tf import d4pg
from acme.tf import networks
from acme.tf import utils as tf2_utils
from acme.utils import loggers
import numpy as np
import sonnet as snt
import gym
import pyvirtualdisplay
import imageio
import base64

# Set up a virtual display for rendering.
# display = pyvirtualdisplay.Display(visible=0, size=(1400, 900)).start()

environment = gym.make("CartPole-v1")
environment = wrappers.GymWrapper(environment)  # To dm_env interface.
environment = wrappers.SinglePrecisionWrapper(environment)

# Grab the spec of the environment.
environment_spec = specs.make_environment_spec(environment)

num_dimensions = np.prod(environment_spec.actions.shape, dtype=int)


# BasicRNN, taken from r2d2 test
# https://github.com/deepmind/acme/blob/master/acme/agents/tf/r2d2/agent_test.py
class BasicRNN(networks.RNNCore):
    def __init__(self, action_spec: specs.DiscreteArray):
        super().__init__(name="basic_r2d2_RNN_network")
        self._net = snt.DeepRNN(
            [
                snt.Flatten(),
                snt.LSTM(20),
                snt.nets.MLP([50, 50, action_spec.num_values]),
            ]
        )

    def __call__(self, inputs, state):
        return self._net(inputs, state)

    def initial_state(self, batch_size: int, **kwargs):
        return self._net.initial_state(batch_size)

    def unroll(self, inputs, state, sequence_length):
        return snt.static_unroll(self._net, inputs, state, sequence_length)


agent_logger = loggers.TerminalLogger(label="agent", time_delta=10.0)
env_loop_logger = loggers.TerminalLogger(label="env_loop", time_delta=10.0)

# Create the D4PG agent.
agent = r2d2.R2D2(
    environment_spec=environment_spec,
    network=BasicRNN(environment_spec.actions),
    burn_in_length=10,
    trace_length=50,
    replay_period=61,
    logger=agent_logger,
    checkpoint=False,
)

# Create an loop connecting this agent to the environment created above.
env_loop = environment_loop.EnvironmentLoop(
    environment, agent, logger=env_loop_logger, should_update=True
)

while True:
    trial_count = 10
    learning_run_count = 100
    trials = [env_loop.run_episode() for i in range(trial_count)]
    total_steps = sum([t["episode_length"] for t in trials])
    print(
        f'{total_steps/len(trials):.2f} after {trials[-1]["episodes"]} episodes'
    )
    env_loop.run(num_episodes=learning_run_count)
