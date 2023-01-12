## Reversi

- Reversible interactions are good - how to leverage it as reward?

- Propose: Multi-agent training:
  - Forward policy generates trajectory
  - Reverse policy tries to reverse it using goal directed reinforcement learning
  - Forward policy gets rewarded based on how well the reverse policy performs on the trajectory
  - Backward policy gets a masked version of forward trajectory -> encourages forward policy to generate trajectories that are easy to reverse?