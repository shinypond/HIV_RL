import numpy as np

lmbd1 = 10000
lmbd2 = 31.98
d1 = 0.01
d2 = 0.01
k1 = 8.0e-7
k2 = 1.0e-4
f = 0.34
delta = 0.7
m1 = 1.0e-5
m2 = 1.0e-5
n_T = 100
c = 13
rho1 = 1
rho2 = 1
lmbd_E = 1
b_E = 0.3
d_E = 0.25
K_b = 100
K_d = 500
delta_E = 0.1
# INIT_STATE = np.log10(np.array([1.0e+6, 3198, 1.0e-4, 1.0e-4, 1, 10], dtype=np.float32))
INIT_STATE = np.log10(np.array([163573, 5, 11945, 46, 63919, 24], dtype=np.float32))

Q = 0.1
R1 = 20000
R2 = 20000
S = 1000

min_a1 = 0.0
max_a1 = 0.7
interval_a1 = 0.7
min_a2 = 0.0
max_a2 = 0.3
interval_a2 = 0.3