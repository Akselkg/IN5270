from wave1D_dn_vc import solver
import numpy as np

"""
 a)
"""

L = 1
C = 0.8  # This could be changed
dt = 0.1
T = 1
m = 9  # number of iterations halving dt.


class Errors:
    """Calculate and store errors of solutions"""

    e2_sum = 0
    errors = []
    def __call__(self, u, x, t, n):
        """Called by solver"""
        if n == len(t) - 1:
            dx = x[1] - x[0]
            dt = t[1] - t[0]
            self.errors = self.errors + [np.sqrt(dx * dt * self.e2_sum)]
            self.e2_sum = 0
            return

        self.e2_sum += sum(u_e(x, t[n]) - u)

errors = Errors()

def q(x):
    return 1 + (x - L/2)**4

def f(x, t):
    pi_L = np.pi / L
    return np.cos(t) * np.cos(pi_L*x) * (
           pi_L**2 * (1 + (x-L/2)**4) + pi_L * np.tan(pi_L*x) * 4*(x-L/2)**3 - 1)

def u_e(x, t):
    return np.cos(np.pi * x/L) * np.cos(t)

def I(x):
    return u_e(x, 0)

# V(x) == u_t(x,0) == 0
def V(x):
    return 0

def c(x):
    return np.sqrt(q(x))


# essentially a modified Wave1D_dn_vc.py implementing different approximations
def neumann_solver(I, V, f, c, L, dt, C, T, user_action=errors, approx='a'):
    Nt = int(round(T/dt))
    t = np.linspace(0, Nt*dt, Nt+1)

    dx = dt * max(c(np.linspace(0, L))) / C
    Nx = int(round(L/dx))
    x = np.linspace(0, L, Nx+1)

    u   = np.zeros(Nx+1)   # Solution array at new time level
    u_1 = np.zeros(Nx+1)   # Solution at 1 time level back
    u_2 = np.zeros(Nx+1)   # Solution at 2 time levels back


    c_ = np.zeros(x.shape)
    for i in range(Nx+1):
        c_[i] = c(x[i])

    q = c_**2

    C2 = (dt/dx)**2; dt2 = dt*dt    # Help variables in the scheme

    Ix = range(0, Nx+1)
    It = range(0, Nt+1)

    # Load initial condition into u_1
    for i in range(0,Nx+1):
        u_1[i] = I(x[i])

    if user_action is not None:
        user_action(u_1, x, t, 0)

    # Special formula for the first step
    for i in Ix[1:-1]:
        u[i] = u_1[i] + dt*V(x[i]) + \
        0.5*C2*(0.5*(q[i] + q[i+1])*(u_1[i+1] - u_1[i]) - \
                0.5*(q[i] + q[i-1])*(u_1[i] - u_1[i-1])) + \
        0.5*dt2*f(x[i], t[0])

    i = Ix[0]
        # Set boundary values (x=0: i-1 -> i+1 since u[i-1]=u[i+1]
        # when du/dn = 0, on x=L: i+1 -> i-1 since u[i+1]=u[i-1])
    ip1 = i+1
    im1 = ip1  # i-1 -> i+1

    if approx == 'a':
        u[i] = u_1[i] + dt*V(x[i]) + \
           0.5*C2 * 2*q[i] * (u_1[im1] - u_1[i]) + dt2 * f(x[i], t[1])
    elif approx == 'b':
        u[i] = u_1[i] + dt*V(x[i]) + \
           0.5*C2 * 2*c(x[i]+0.5*dx)**2 * (u_1[im1] - u_1[i]) + dt2 * f(x[i], t[1])
    # elif approx == 'c':


    i = Ix[-1]
    im1 = i-1
    ip1 = im1  # i+1 -> i-1

    if approx == 'a':
        u[i] = u_1[i] + dt*V(x[i]) + \
           0.5*C2 * 2*q[i] * (u_1[im1] - u_1[i]) + dt2 * f(x[i], t[1])
    elif approx == 'b':
        u[i] = u_1[i] + dt*V(x[i]) + \
           0.5*C2 * 2*c(x[i]-0.5*dx)**2 * (u_1[im1] - u_1[i]) + dt2 * f(x[i], t[1])

    if user_action is not None:
        user_action(u, x, t, 1)

    u_2, u_1, u = u_1, u, u_2

    for n in It[1:-1]:

        for i in Ix[1:-1]:
            u[i] = - u_2[i] + 2*u_1[i] + \
                C2*(0.5*(q[i] + q[i+1])*(u_1[i+1] - u_1[i])  - \
                    0.5*(q[i] + q[i-1])*(u_1[i] - u_1[i-1])) + \
            dt2*f(x[i], t[n])

        # Set boundary values, using (54)
        # x=0: i-1 -> i+1 since u[i-1]=u[i+1] when du/dn=0
        # x=L: i+1 -> i-1 since u[i+1]=u[i-1] when du/dn=0
        i = Ix[0]
        ip1 = i+1
        im1 = ip1
        if approx == 'a':
            u[i] = - u_2[i] + 2*u_1[i] + \
                    C2 * 2*q[i] * (u_1[im1] - u_1[i]) + dt2 * f(x[i], t[n])
        elif approx == 'b':
            u[i] = - u_2[i] + 2*u_1[i] + \
                    C2 * 2*c(x[i]+0.5*dx)**2 * (u_1[im1] - u_1[i]) + dt2 * f(x[i], t[1])


        i = Ix[-1]
        im1 = i-1
        ip1 = im1
        if approx == 'a':
            u[i] = - u_2[i] + 2*u_1[i] + \
                    C2 * 2*q[i] * (u_1[im1] - u_1[i]) + dt2 * f(x[i], t[n])
        elif approx == 'b':
            u[i] = - u_2[i] + 2*u_1[i] + \
                    C2 * 2*c(x[i]-0.5*dx)**2 * (u_1[im1] - u_1[i]) + dt2 * f(x[i], t[1])

        if user_action is not None:
            if user_action(u, x, t, n+1):
                break

        # Update data structures for next step
        #u_2[:] = u_1;  u_1[:] = u  # safe, but slower
        u_2, u_1, u = u_1, u, u_2

    # Important to correct the mathematically wrong u=u_2 above
    # before returning u
    u = u_1
    return


for i in range(0, m):
    neumann_solver(I, V, f, c, L, dt, C, T, user_action=errors, approx='a')
    if i == 0:
        continue

    r = np.log(errors.errors[i] / errors.errors[i-1]) / np.log(0.5)
    print(i, r)
    dt = 0.5 * dt

"""
output:
(1, -0.0)
(2, 1.3485323891936725)
(3, 1.0720058441433811)
(4, 1.117077244536596)
(5, 1.0569354162502986)
(6, 1.0063808634881566)
(7, 1.0097371341220909)
(8, 1.0021242254536409)

showing a convergence rate tending to 1. Linear convergence.
"""

"""
 b)
"""
print('b)')


def f_b(x, t):
    pi_L = np.pi / L
    return -np.cos(t) * np.cos(pi_L*x) + pi_L**2 * np.cos(t)*(np.cos(pi_L*x) + np.cos(2*pi_L*x))

def q_b(x):
    return 1 + np.cos(np.pi/L * x)

def c_b(x):
    return np.sqrt(q_b(x))

b_errors = Errors()

for i in range(0, m):
    neumann_solver(I, V, f_b, c_b, L, dt, C, T, user_action=errors, approx='b')
    if i == 0:
        continue

    r = np.log(b_errors.errors[i] / b_errors.errors[i-1]) / np.log(0.5)
    print(i, r)
    dt = 0.5 * dt

"""
not sure why implementation in b) is not working yet.
"""
