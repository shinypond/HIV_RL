cfg = {}


# Environment Configuration
max_days = 1000
treatment_days = 5
reward_scaler = 1e+8 # IMPORTANT
MAX_EPISODE_STEPS = max_days // treatment_days 


# Agent Configuration
cfg['dqn_agent'] = dict(
    max_days = max_days,
    treatment_days = treatment_days,
    reward_scaler = reward_scaler,
    memory_size = int(1e6),
    batch_size = 2048,
    lr = 2e-4,
    l2_reg = 0.,
    grad_clip = 1000.,
    target_update = MAX_EPISODE_STEPS * 5,
    max_epsilon = 1.0,
    min_epsilon = 0.05,
    epsilon_decay = 1 / 200,
    decay_option = 'logistic',
    discount_factor = 0.99,
    n_train = 1,
    hidden_dim = 1024,
    per = True, # IMPORTANT
    alpha = 0.2,
    beta = 0.6,
    beta_increment_per_sampling = 5e-6,
    prior_eps = 1e-6,
    double_dqn = True, # IMPORTANT
)

