import numpy as np
import matplotlib.pyplot as plt

def cooling(T0, k, T_s, t_end, dt, theta=0.5):
    N = int(np.ceil(t_end/dt))
    T = np.zeros(N)
    T[0] = T0
    for n in range(0, N-1):
        T[n+1] = ((k*T_s(dt*n) + (theta-1)*k*T[n])*dt + T[n])/(1 + k*theta*dt)

    plt.plot(np.linspace(0, t_end, N), T)
    return T

def T_s(t):
    return 1

cooling(10, 0.5, T_s, 10, 0.1)
plt.plot(np.linspace(0,10,50), np.ones(50))
plt.show()
