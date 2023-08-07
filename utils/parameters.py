parameters = {
    "episodes": 600,
    "max_steps": 1000,
    "epsilon": 1,
    "epsilon_decay": 0.99,
    "epsilon_min": 0.01,
    "update_freq": 4,
    "gamma": 0.999,
    "seed": 12,
    "lr": 0.0005,
    "tau": 0.001,
    "buffer_size": 100_000,
    "batch_size": 64,
    "reward_target_mean": 2000,
    "render": True,
    "record": False      # render must be true as well
}