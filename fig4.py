import scipy
import ecosim
import numpy as np
import matplotlib.pyplot as plt

def generate_structure_matrix(alpha,S):
    
    intensity = np.random.power(2,S)
    
    structure_matrix = intensity[:,None] @ np.ones((1,S))
    return alpha * structure_matrix  / S

def simulate(str_mat,sig,S):

    glv_model = ecosim.glv.LotkaVolterra(S,trans='log')
    glv_model.mat_alpha = str_mat + np.random.normal(0,sig/np.sqrt(S),(S,S)) + np.identity(S)

    glv_model.vec_x = np.random.uniform(0,1,S)
    glv_model.K = 1
    glv_model.r = 1
    glv_model.lam = 1e-8
    
    traj = glv_model.run(250,1)
    return traj['last']

def solve_cavity(str_mat,sig,n_iter = 2000,r=.25):
    
    S = len(str_mat)
    
    omega0 = lambda delta: (1 + scipy.special.erf(delta / np.sqrt(2))) / 2
    omega1 = lambda delta: delta * omega0(delta) + np.exp(-(delta**2) / 2) / np.sqrt(2 * np.pi)
    omega2 = lambda delta: omega0(delta) * (1 + delta**2) + delta * np.exp(-(delta**2) / 2) / np.sqrt(2 * np.pi)

    omega0 = np.vectorize(omega0)
    omega1 = np.vectorize(omega1)
    omega2 = np.vectorize(omega2)

    # We compute the eigenvalues and eigenvectors of the interaction matrix
    # and filter them
    thr = 1e-2
    u,eig,v = np.linalg.svd(str_mat)
    n_eig = len(eig[eig>thr])
    u = u[:,:n_eig] * eig[:n_eig] * np.sqrt(S)
    v = v[:n_eig,:] / np.sqrt(S)
    
    #beta is a species index and i is an eigenvalue index
    delta = lambda beta,f,q : (1 - f @ u[beta,:])/np.sqrt(q*sig**2)
    new_f = lambda d,q: np.sqrt(q)*sig *np.array([v[i,:] @ omega1(d) for i in range(n_eig)])
    new_q = lambda d,q : sig**2 *q * np.mean(omega2(d))
    
    f0 = np.random.uniform(-1,1,n_eig)
    q0 = 1

    for i in range(n_iter):
        d = np.array([delta(beta,f0,q0) for beta in range(S)])
        
        f0 = f0 * (1-r) + r*new_f(d,q0)
        q0 = q0 * (1-r) + r*new_q(d,q0)
        
    print(f"Errors are {np.abs(f0-new_f(d,q0))} and {np.abs(q0-new_q(d,q0))}")

    sp_means = np.sqrt(q0)*sig * d
    std = sig * np.sqrt(q0)

    th_sad = lambda x : np.mean(np.exp(-(x-sp_means)**2 / (2 * std**2)))/np.sqrt(2 * np.pi * std**2)
    return np.vectorize(th_sad)

def solve(alpha,sig,S):
    
    str_mat = generate_structure_matrix(alpha,S)
    th_sad = solve_cavity(str_mat,sig)
    sim_sad = simulate(str_mat,sig,S)
        
    return [sim_sad,th_sad]

def plot(sim_sads,th_sads):

    n_plots = len(sim_sads)
    fig, ax = plt.subplots(1,n_plots,figsize=(10*n_plots,6))
    if n_plots == 1:
        ax = [ax]

    for i, data in enumerate(zip(sim_sads,th_sads)):

        # Eliminate upper and right axes
        ax[i].spines['right'].set_color('none')
        ax[i].spines['top'].set_color('none')

        # Show ticks in the left and lower axes only
        ax[i].xaxis.set_ticks_position('bottom')
        ax[i].yaxis.set_ticks_position('left')
        
        # Limit the number of ticks
        ax[i].set_yticks([0,1,2])
        ax[i].set_xticks(np.arange(0,max(data[0])*1.1,.5))

        ax[i].hist(data[0],density=True,color='#97BDF4')

        x_range = np.linspace(0,max(data[0])*1.1,100)
        y_data = data[1](x_range)
        
        ax[i].plot(x_range,y_data,color='#553A7F',alpha=1,linewidth=6)
        ax[i].set_xlabel(f'$x_i^\star$')    
    
    plt.show()
    
def main():

    S = 1000
    alphas = [3,3]
    sigs = [.25,.75]

    sim_sads,th_sads = [],[]

    for alpha,sig in zip(alphas,sigs):
        data = solve(alpha,sig,S)
        sim_sads.append(data[0])
        th_sads.append(data[1])

    plot(sim_sads,th_sads)

if __name__ == '__main__':
    main()
    