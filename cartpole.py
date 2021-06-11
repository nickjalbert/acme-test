# https://github.com/deepmind/acme/blob/master/examples/quickstart.ipynb

from acme import environment_loop
from acme import specs
from acme import wrappers
from acme.agents.tf import r2d2
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
#display = pyvirtualdisplay.Display(visible=0, size=(1400, 900)).start()

environment = gym.make('CartPole-v1')
environment = wrappers.GymWrapper(environment)  # To dm_env interface.
environment = wrappers.SinglePrecisionWrapper(environment)

# Grab the spec of the environment.
environment_spec = specs.make_environment_spec(environment)

