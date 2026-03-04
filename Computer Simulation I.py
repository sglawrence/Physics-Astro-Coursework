#!/usr/bin/python3 

# required modules do not change

import time
import numpy as np
from scipy import linalg
from scipy.sparse import csc_matrix
from scipy.sparse.linalg import cg
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm

# initial conditions

size = 500
maxit = 24000
div = 1000
k = 0.02 # implicit algorithm stable beyond k = 0.02
h = 0.2
r = k / h / h

# initialize arrays

V = np.zeros(shape = (int(maxit / div), size))
u = np.zeros(size)
v = np.zeros(size)
u_explicit = np.zeros(shape = (size, 2))

# set initial condition

for j in range(size):
    
    u[j] = u_explicit[j][0] = np.sin(0.5 * 2 * np.pi * (j + 1) / (size + 1))

# explicit method


t = 0
start_time_explicit = time.time()


for i in range(1, maxit):
    
    for j in range(1, size - 1):
        
        u_explicit[j][1] = u_explicit[j][0] + r * (u_explicit[j+1][0] + u_explicit[j-1][0] - 2.0 * u_explicit[j][0])
        
        if (i % div == 0):
            
            V[t][j] = u_explicit[j][1]
            
    for k in np.arange(0, size):
        
            u_explicit[k][0] = u_explicit[k][1]
            
    if (i % div == 0):
        
        t += 1   
            
    u_explicit[:, 0] = u_explicit[:, 1]

end_time_explicit = time.time()
time_explicit = end_time_explicit - start_time_explicit

# initialise matrix

M = np.eye(size) * (1 + 2 * r)
M += np.eye(size, k=-1) * -r
M += np.eye(size, k=1) * -r
LU = linalg.lu_factor(M)
u_implicit = np.zeros((int(maxit / div), size))
u_cg = np.zeros((int(maxit / div), size))
u_scg = np.zeros((int(maxit / div), size))

# implicit method

u = np.sin(0.5 * 2 * np.pi * np.arange(1, size + 1) / (size + 1)) # resetting initial condition

start_time_implicit = time.time()


for i in range(maxit):
    
    b = u.copy()
    
    b[0] = b[-1] = 0
    
    u = linalg.lu_solve(LU, b)
    
    if i % div == 0:
        
        u_implicit[i // div] = u
        
    
end_time_implicit = time.time()
time_implicit = end_time_implicit - start_time_implicit


# conjugate_gradient function


def conjugate_gradient(A, b, x=None):
    
    n = len(b)
    
    if not x:
        
        x = np.zeros_like(b)
        
    r = np.dot(A, x) - b
    
    p = - r
    
    r_k_norm = np.dot(r, r)
    
    for i in range(2 * n):
        
        Ap = np.dot(A, p)
        
        alpha = r_k_norm / np.dot(p, Ap)
        
        x += alpha * p
        
        r += alpha * Ap
        
        r_kplus1_norm = np.dot(r, r)
        
        beta = r_kplus1_norm / r_k_norm
        
        r_k_norm = r_kplus1_norm
        
        if r_kplus1_norm < 1e-5:
            
            break
        
        p = beta * p - r
        
    return x


# (ii) conjugate gradient method

u = np.sin(0.5 * 2 * np.pi * np.arange(1, size + 1) / (size + 1)) # resetting initial condition

start_time_cg = time.time()


for i in range(maxit):
    
    b = u.copy()
    
    b[0] = b[-1] = 0
    
    u = conjugate_gradient(M, b)
    
    if i % div == 0:
        
        u_cg[i // div] = u


end_time_cg = time.time()
time_cg = end_time_cg - start_time_cg


# (iii) sparse conjugate gradient method

u = np.sin(0.5 * 2 * np.pi * np.arange(1, size + 1) / (size + 1)) # resetting initial condition

A_sparse = csc_matrix(M)

start_time_scg = time.time()

for i in range(maxit):
    
    b = u.copy()
    
    b[0] = b[-1] = 0
    
    u, _ = cg(A_sparse, b)
    
    if i % div == 0:
        
        u_scg[i // div] = u


end_time_scg = time.time()
time_scg = end_time_scg - start_time_scg


# print times

print("time for explicit method: ", time_explicit)
print("time for implicit method: ", time_implicit)
print("time for conjugate gradient method: ", time_cg)
print("time for sparse conjugate gradient method: ", time_scg)

# plotting

x = np.arange(0, size)
t = np.arange(0, maxit, div)
X, T = np.meshgrid(x, t)
J = np.arange(0, size, 1)
K = np.arange(0, int(maxit/div), 1)
J, K = np.meshgrid(J, K)

fig = plt.figure(figsize=(10, 16), dpi=300)

# explicit method plot

ax1 = fig.add_subplot(4, 1, 1, projection='3d')
surf = ax1.plot_surface(J, K*1000, V, cmap=cm.viridis, linewidth=0, antialiased=False)
ax1.set_xlabel('Size', fontsize = 12, labelpad=10)
ax1.set_ylabel('Time', fontsize = 12, labelpad=12)
ax1.set_zlabel('Temperature', fontsize = 12, labelpad=10)
ax1.set_title(f'Explicit Method (Time: {time_explicit:.2f} [s])', fontsize = 15, fontweight='normal', loc = 'left')

# implicit method plot

ax2 = fig.add_subplot(4, 1, 2, projection='3d')
surf = ax2.plot_surface(X, T, u_implicit, cmap=cm.viridis, linewidth=0, antialiased=False)
ax2.set_xlabel('Size', fontsize = 12, labelpad=10)
ax2.set_ylabel('Time', fontsize = 12, labelpad=12)
ax2.set_zlabel('Temperature', fontsize = 12, labelpad=10)
ax2.set_title(f'Implicit Method (Time: {time_implicit:.2f} [s])', fontsize = 15, fontweight='normal', loc = 'left')

# conjugate gradient method plot

ax3 = fig.add_subplot(4, 1, 3, projection='3d')
surf = ax3.plot_surface(X, T, u_cg, cmap=cm.viridis, linewidth=0, antialiased=False)
ax3.set_xlabel('Size', fontsize = 12, labelpad=10)
ax3.set_ylabel('Time', fontsize = 12, labelpad=12)
ax3.set_zlabel('Temperature', fontsize = 12, labelpad=10)
ax3.set_title(f'Conjugate Gradient Method (Time: {time_cg:.2f} [s])', fontsize = 15, fontweight='normal', loc = 'left')

# sparse conjugate gradient method plot

ax4 = fig.add_subplot(4, 1, 4, projection='3d')
surf = ax4.plot_surface(X, T, u_scg, cmap=cm.viridis, linewidth=0, antialiased=False)
ax4.set_xlabel('Size', fontsize = 12, labelpad=10)
ax4.set_ylabel('Time', fontsize = 12, labelpad=12)
ax4.set_zlabel('Temperature', fontsize = 12, labelpad=10)
ax4.set_title(f'Sparse Conjugate Gradient Method (Time: {time_scg:.2f} [s])', fontsize = 15, fontweight='normal', loc = 'left')

plt.tight_layout()
plt.subplots_adjust(hspace=0.2)
plt.show()



# (iv) comparing the efficiency of LU decomposition vs. conjugate gradient 'cg' method

# initialize the range of matrix sizes

sizes = np.arange(250, 2500, 50) # I took some liberties here

lu_times = []
cg_times = []

for size in sizes:
    
    # using random matrix and vector [same case as previous matrices, doesn't impact result]
    
    R = np.random.rand(size, size)
    r = np.random.rand(size)

    # LU decomposition
    
    start = time.time()
    
    P, L, U = linalg.lu(R)
    lu_solve = linalg.lu_solve(linalg.lu_factor(R), r)
    
    end = time.time()
    lu_times.append(end - start)

    # conjugate gradient
    
    R_sparse = csc_matrix(R)
    
    start = time.time()
    
    x, info = cg(R_sparse, r)
    
    end = time.time()
    cg_times.append(end - start)


# (v) gaussian initial conditions and comparison with greens function for diffusion


# parameters

size = 1000
maxit = 24000
div = 100 # changed from 1000 just so graph is less choppy :)
k = 0.02
h = 0.2
r = k / h / h

# initialize arrays

V = np.zeros((int(maxit / div), size))
u = np.zeros(size)
u_explicit = np.zeros((size, 2))

# gauss initial condition

u = u_explicit[:, 0] = np.exp(-4000 * (np.arange(size) - size / 2)**2 / size**2)

# normalise the initial condition

u /= np.trapz(u, dx=h)

# initialize matrix

M = np.eye(size) * (1 + 2 * r)
M += np.eye(size, k=-1) * -r
M += np.eye(size, k=1) * -r
LU = linalg.lu_factor(M)
u_implicit = np.zeros((int(maxit / div), size))

# LU decomposition method

for i in range(maxit):
    
    b = u.copy()
    
    b[0] = b[-1] = 0
    
    u = linalg.lu_solve(LU, b)
    
    if i % div == 0:
        
        u_implicit[i // div] = u

# parameters for propagator solution / greens function

D = k / h / h # diffusion coefficient

x = size / 2 # centre point

t_val = np.arange(1, maxit / div + 1) # time values


y_values = np.linspace(-size/2, size/2, size) # value grid

u_naught_values = np.exp(-4000 * (y_values - size / 2)**2 / size**2) # initial condition

u_naught_values /= np.trapz(u_naught_values, dx=h) # initial condition normalisation

sigma_naught = size / (2 * np.sqrt(2 * np.log(2))) # width of the gaussian


u_propagator = np.zeros((int(maxit / div), size))

for i in range(int(maxit / div)):
    
    for j in range(size):
        
        x_p = j
        
        u_propagator[i, j] = np.exp(-(x - x_p)**2 / (4 * D * t_val[i])) / np.sqrt(20 * np.pi * D * t_val[i])

# plotting the propagator solution alongside the numerical solution

fig, axs = plt.subplots(1, 2, figsize=(20, 6), dpi=300)
gap = 10
c = 0.7
ftsz = 15
tick_size = 13

# gaussian initial conditions plot

axs[1].plot(t_val, u_implicit[:, size // 2], label='numerical solution', alpha=c)
axs[1].plot(t_val, u_propagator[:, size // 2], label='propagator solution', alpha=c, linestyle='--', color='k')
axs[1].set_title(r'Numerical and Propagator Solutions at $\left(j = size / 2 \right)$ versus Time', y = 1.15 , fontsize=(ftsz+2))
axs[1].set_xlabel('Iteration [n]', labelpad=gap, fontsize=ftsz)
axs[1].set_ylabel('Solution', labelpad=gap, fontsize=ftsz)
axs[1].tick_params(axis='both', which='major', labelsize=tick_size)
axs[1].legend(fontsize = ftsz)
axs[1].grid(True)

# efficiency comparison plot

axs[0].plot(sizes, lu_times, label='LU decomposition', alpha=c, color='k', linestyle='--')
axs[0].plot(sizes, cg_times, label='conjugate gradient', alpha=c, color='mediumorchid', linestyle='-')
axs[0].set_title('Efficiency of LU Decomposition versus Conjugate Gradient', y = 1.15, fontsize=(ftsz+2))
axs[0].set_xlabel('Size [n]', labelpad=gap, fontsize=ftsz)
axs[0].set_ylabel('Time [seconds]', labelpad=gap, fontsize=ftsz)
axs[0].tick_params(axis='both', which='major', labelsize=tick_size)
axs[0].legend(fontsize = ftsz)
axs[0].grid(True)

plt.show()