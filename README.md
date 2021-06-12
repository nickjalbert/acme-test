# Acme Test

Initial installation and exploration of DeepMind's [Acme RL
Framework](https://github.com/deepmind/acme).


* [Acme Docs](https://github.com/deepmind/acme/tree/master/docs)
* [R2D2 for discrete action control](https://github.com/deepmind/acme/tree/master/acme/agents/tf/r2d2)
* [Quickstart and Tutorial](https://github.com/deepmind/acme/tree/master/examples)

## Troubles

* [Reverb](https://github.com/deepmind/reverb) (a DB for RL) is only available
  for "linux like" OSes.

    * This is interesting: a lot of the frameworks we may be building against
      aren't necessarily cross-platform.  How to handle situations like this,
      esp. if we imagine AgentOS as a cross-platform endeavor.


## Notes

* I think we'd probably call the Acme agent a policy in our terminology

* Gameplan:
    * Create an R2D2 agent to play Cartpole in pure Acme
    * Create an X agent to play Cartpole in pure RLlib
    * Create an AOS Cartpole agent
    * Start porting

* To think about: we're solving a problem that we really don't have.  We think
  other people might have it (but it's still early days). One benefit of
  building the hybrid agent is that it forces us to have the problem we're
  trying to solve.  A way forward: build a *crappy* hybrid agent just to expose
  ourselves to the composability problem.



## Command

```
virtualenv -p /usr/local/opt/python\@3.8/bin/python3.8 acme-test
source acme-test/bin/activate
pip install --upgrade pip setuptools
pip install dm-acme
pip install dm-acme[reverb]  # Fails with ERROR: ResolutionImpossible
pip install dm-reverb # Fails.  Only support linux
pip install dm-acme[envs]
pip install tensorflow
pip install tensorflow-probability
pip install dm-sonnet
pip install imageio PILLOW pyvirtualdisplay
```

## Docker notes

```
./docker/build # build latest Dockerfil
./docker/run   # run a container from latest build
./docker/exec  # interactive bash prompt into latest container
./docker/kill  # kill running container
```
