# tail_length_random
Map size: 12x12
Trained on random tail lengths [0, 10]
Maps: empty 1M timesteps, increase {0.0, ..., 0.09} * 100k timesteps, no-increase 0.09 1M timesteps
Reward: manhattan for all. 10 for fruit, -10 for death

# tail_length_3
Map size: 12x12
Maps: empty 1M timesteps (all)
Reward: manhattan, negative (-0.01/timestep), no reward. 10 for fruit, -10 for death

# tail_length_0
Map size: 12x12
Tail starts at length 0
Maps: empty 1M timesteps, increase {0.0, ..., 0.09} * 100k timesteps, no-increase 0.09 1M timesteps
Reward: manhattan for all. 10 for fruit, -10 for death
