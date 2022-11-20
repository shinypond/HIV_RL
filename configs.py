cfg = {}

# Agent Configuration

cfg['rainbow_agent'] = dict(
    # Training parameters
    memory_size = int(1e6),
    batch_size = 1024,
    lr = 1e-4,
    l2_reg = 0.,
    target_update = 600 * 10,
    gamma = 1.,
    # Network parameters
    hidden_dim = 512,
    dropout = 0.0,
    # PER parameters
    alpha = 0.2,
    beta = 0.6,
    prior_eps = 1e-6,
    # Categorical DQN parameters
    v_min = 0.,
    v_max = 100.,
    atom_size = 51,
    # N-step Learning parameters
    n_step = 1,
)

cfg['perdqn_agent'] = dict(
    memory_size = int(1e6),
    batch_size = 2048,
    lr = 1e-4,
    l2_reg = 0.,
    grad_clip = 10.,
    target_update = 600 * 10,
    max_epsilon = 1.0,
    min_epsilon = 0.05,
    epsilon_decay = 1 / 2000,
    hidden_dim = 512,
    alpha = 0.2,
    beta = 0.6,
    prior_eps = 1e-6,
    double_dqn = True,
)
