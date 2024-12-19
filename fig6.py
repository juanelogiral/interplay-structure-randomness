import ecosim
import numpy as np
import matplotlib.pyplot as plt
import colorsys
import pickle

def generate_str_matrix(mu,beta,nf,s):

    # Number of species
    S = nf * s

    # Variabilities
    beta  = .3
    mu = .5

    # group matrix
    grp_mat = np.random.normal(mu,beta,(nf,nf))
    grp_mat -= np.diag(np.diag(grp_mat))
    grp_mat = np.kron(grp_mat,np.ones((s,s)))/s
    return grp_mat

def generate_vec_init(nf,s):
    return np.kron(np.random.uniform(0,1,nf),np.ones(s))

def simulate(int_matrix,vec_init):
    
    S = len(int_matrix)
    glv = ecosim.glv.LotkaVolterra(S,trans='log')
    glv.mat_alpha = int_matrix + np.identity(S)
    glv.K=1
    glv.r=1
    glv.lam=1e-8
    glv.vec_x = vec_init

    try:
        traj = glv.run(4000,4)
    except ecosim.base.DivergenceError as e:
        traj = e.traj
        print("Divergence")

    return traj

def calculate_variation_amplitude(traj,grp_traj):

    mat_x = traj[2500:].mat_x
    mat_grp_x = grp_traj[2500:].mat_x

    sp_var = mat_x.max(axis=0) - mat_x.min(axis=0)
    grp_var = mat_grp_x.max(axis=0) - mat_grp_x.min(axis=0)

    return np.max(sp_var),np.max(grp_var)

def randcolors(n_group,n_per_group,seed=None,type='bright',color_offset=None):
    """Returns ``n_group`` * ``n_per_group`` random colours such that colors in a group are coherent
    type: {bright, soft}
    """
    color_rng = np.random.default_rng(seed)

    if type not in ("bright","soft"):
        raise ValueError("Please choose 'bright' or 'soft' for type")

    if type =='bright':
        low,high = .9,1
    elif type=='soft':
        low,high=.5,.8
    # Generate color map for bright colors, based on hsv
    if color_offset is None:
        color_offset = .2
    grp_HV = [
        (
            (.9*i/n_group + color_offset) % 1,
            color_rng.uniform(low=low, high=high),
        )
        for i in range(n_group)
    ]
    eps = .2/n_group 
    colors_grp_HSV = [((h+np.random.uniform(-eps,eps))%1,.75,v) for h,v in grp_HV]
    colors_HSV = [((h+np.random.uniform(-eps,eps))%1,s,v) for h,sg,v in colors_grp_HSV for s in np.concatenate([[.9],np.random.uniform(.4,.9,n_per_group-1)])]

    colors_RGB = []
    colors_grp_RGB = []
    for c in colors_HSV:
        colors_RGB.append(colorsys.hsv_to_rgb(c[0], c[1], c[2]))
    for c in colors_grp_HSV:
        colors_grp_RGB.append(colorsys.hsv_to_rgb(c[0], c[1], c[2]))

    return colors_grp_RGB,colors_RGB


def plot(sig_list,traj_list,plot_idx,nf,s):

    traj_grp_list = np.array([traj.map(lambda x : x.reshape(nf,-1).mean(axis=1)) for traj in traj_list])
    sp_var,grp_var = np.transpose([calculate_variation_amplitude(traj,traj_grp) for traj, traj_grp in zip(traj_list,traj_grp_list)])

    grp_colors,colors = randcolors(nf,s)
    print(len(plot_idx))
    fig,ax = plt.subplots(len(plot_idx)+1,2,figsize=(14,2.5+len(plot_idx)*2.5))
    fig.tight_layout(pad=1)

    for i,(sig,traj,traj_grp) in enumerate(zip(sig_list[plot_idx],traj_list[plot_idx],traj_grp_list[plot_idx])):
        
        ax[i,0].stackplot(traj_grp[10:].vec_t,traj_grp[10:].mat_x.T,colors = grp_colors)
    
        vec_t,mat_x = traj[10:].anchors
        for j in range(0,nf*s,s//3):
            ax[i,1].plot(vec_t,mat_x[:,j],color = colors[j],alpha=1)
        
        ax[i,0].set_xlim(10,traj.T)
        ax[i,0].set_yticks([0,1,2,3])
        ax[i,1].set_xlim(10,traj.T)
        ax[i,1].set_yticks([0,.3,.6,.9] if np.max(mat_x) > .8 else [0,.3,.6])
        ax[i,1].set_ylim(0,np.max(mat_x))
        if i < len(traj_list) - 1:
            ax[i,0].get_xaxis().set_visible(False)
            ax[i,1].get_xaxis().set_visible(False)
        else:
            ax[i,0].set_xticks([0,1000,2000,3000,4000])
            ax[i,1].set_xticks([0,1000,2000,3000,4000])
            
        ax[i,0].tick_params(axis='both',labelsize=15)
        ax[i,1].tick_params(axis='both',labelsize=15)
        
        if i == len(traj_list)-1:
            ax[i,0].set_xlabel("Time")
            ax[i,1].set_xlabel("Time")
        ax[i,0].set_ylabel(r"$f_i(t)$")
        ax[i,1].set_ylabel(r"$x_i(t)$")


    gs = ax[-1, 1].get_gridspec()
    # merge the axes in the last row
    for ax in ax[-1]:
        ax.remove()
    axbig = fig.add_subplot(gs[-1,0:])


    line_inter, = axbig.plot(np.sort(sig_list),grp_var, color = 'blue')
    axbig.scatter(np.sort(sig_list),grp_var,color = 'blue',s=60)

    line_intra, = axbig.plot(np.sort(sig_list),sp_var, color = 'orange')
    axbig.scatter(np.sort(sig_list),sp_var,color = 'orange',s=60)

    axbig.vlines(sig_list[plot_idx],-.1,1.1,linestyles='--',colors='black')
    axbig.set_ylim(0,1.1)
    axbig.set_xlim(-.1,2.8)
    axbig.set_yticks([0,.5,1])
    axbig.set_xticks([0,.5,1,1.5,2,2.5])
    axbig.tick_params(axis='both',labelsize=15)
    axbig.set_xlabel(r"$\sigma$")
    axbig.set_ylabel("Fluctuation""\n""amplitude")

    axbig.legend(handles = [line_inter,line_intra],labels = ['Species level','Strain level'],fontsize = 16,loc = (.8,.45))

    plt.show()

def main():

    sig_list = np.array([0,1,1.25,1.35,1.5,1.65,1.75,1.85,2,2.25,2.5,2.75])
    plot_idx = [0,1,6,11]

    nf,s =  100,20

    try:
        with open('./fig6_files.pkl','rb') as file:
            str_mat,rand_mat,vec_init = pickle.load(file)
        print("Matrices loaded.")
    except:

        str_mat = generate_str_matrix(.5,.3,nf,s)
        rand_mat = np.random.normal(0,1/np.sqrt(nf*s),(nf*s,nf*s))
        vec_init = generate_vec_init(nf,s)

        with open('./fig6_files.pkl','wb') as file:
            pickle.dump([str_mat,rand_mat,vec_init],file)
        print("Matrices not found. Generating new matrices")

    traj_list = []
    for sig in sig_list:
        traj = simulate(str_mat + sig*rand_mat,vec_init)
        traj_list.append(traj)
        print(f"Done {sig}.")

    traj_list = np.array(traj_list)
    plot(sig_list,traj_list,plot_idx,nf,s)


if __name__ == '__main__':
    main()
    
    