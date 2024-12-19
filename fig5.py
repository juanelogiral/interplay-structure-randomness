import matplotlib.pyplot as plt

from scipy.special import erf
from scipy.optimize import root
import matplotlib.colors as mcolors
import pandas as pd
import numpy as np

import ecosim


def matplot(mat,vmax=None):
    if vmax is None:
        vmax=np.max(np.abs(mat))
    plt.imshow(mat,vmin=-vmax,vmax=vmax,cmap='seismic_r')
    plt.xticks([],[])
    plt.yticks([],[])
    
    
def omega0(z):
    return (1 + erf(z/np.sqrt(2)))/2
    
def omega1(z):
    return (np.exp(-z**2/2) + np.sqrt(np.pi/2) *z *(1 + erf(z/np.sqrt(2))))/np.sqrt(2*np.pi)
    
def omega2(z):
    return  (np.exp(-z**2/2)*z + np.sqrt(np.pi/2) *(z**2+1) *(1 + erf(z/np.sqrt(2))))/np.sqrt(2*np.pi)
    
    
def solstruct(f,u,v,S):
    return f - 1./S * np.dot(v.T,np.maximum(0,1+ np.dot(u,f) ))
    
def alive(l):
    if np.max(l) < 1e-5:
        return 0
    elif np.min(l) > 1e-2:
        return 1
    thr = np.sort(l)[np.argmax(np.diff(np.log(np.sort(l))))]
    return np.count_nonzero(l>thr)/len(l)
    
def phi(f,q,u,sigma):
    return np.mean(omega0((1+ np.dot(u,f) ) / ( np.sqrt(q)*sigma  )))
    
def solsigma(y,u,v,sigma,S):
    f=y[:-1]
    q=y[-1]
    Delta =  (1+ np.dot(u,f) ) / ( np.sqrt(q)*sigma  )
    det= f - np.sqrt(q)*sigma/S * np.dot(v.T,omega1(Delta))
    rnd= 1 - sigma**2 /S * np.sum(omega2(Delta))
    #print(sigma,rnd,q,sigma**2 /S * np.sum(omega2(Delta)),np.sum(omega2(Delta)),np.mean(f))
    return np.concatenate([det,[rnd]])
    

def comp(us,vs,ax):
    S,nf=us.shape
    sigmas=np.linspace(0.01,1.4,12)
    table=[]
    mat=np.random.normal(0,1,(S,S)) 
    #plt.figure(figsize = (20,20))
    sidx=0
    for s in sigmas:
        locmat=-np.dot(us,vs.T)/S +s*mat/np.sqrt(S) + np.identity(S)
        locmat_red = -np.dot(us,vs.T)[::15,::15]/50 +s*mat[:100,:100]/np.sqrt(100) + np.identity(100)
        
        glv = ecosim.glv.LotkaVolterra(S,trans='log')
        glv.mat_alpha = locmat
        glv.vec_x = np.random.uniform(0,1,S)
        glv.K=1
        glv.r=1
        glv.lam =1e-10
        
        simu=glv.run(1000,1)
        xs=simu['last']
        #plt.plot(np.log10(simu.y.T))
        #plt.show()
        simufs=np.dot(vs.T,xs)/S
        if s==0:
            f0=np.random.random(nf)
            sol=root(solstruct,f0,args=(us,vs,S))
            q=0
        else:
            y0=np.random.random(nf+1)
            if s!= 0.01 and sol.success:
                y0[:len(sol.x)]=sol.x
            sol =root(solsigma,y0,args=(us,vs,s,S))
            q=sol.x[-1]
        if not sol.success:
            res=np.nan*np.ones(nf)
            print("Damn")
            raise Exception("DAMN")
        else:
            res=sol.x[:nf]
        print(s,sol.x,simufs)
        
        dic={'sigma':s,'q':q,'alive':tuple([alive(xs[vs[:,i]>0]) for i in range(nf)] + [alive(xs)]),'phi':tuple([phi(res,q,us[vs[:,i]>0],s) for i in range(nf)] + [phi(res,q,us,s)]) }
        for i, r in enumerate(res):
            dic['theo_{}'.format(i)]=r
        for i, r in enumerate(simufs):
            dic['simu_{}'.format(i)]=r
        table.append(dic)
        
        if s == sigmas[0] or s == sigmas[-1]:
            
            
            #plt.subplot(3,3,sidx),plt.title(r'$\sigma=${:.2g}'.format(s),fontsize=22)
            ax[sidx].tick_params(axis='both', which='major', labelsize=24)
            if s>1.:
                bins=np.linspace(0,6,20)
            else:
                bins=np.linspace(0,6,20)
            colors=list(mcolors.TABLEAU_COLORS)
            for i in range(nf):
                ax[sidx].hist(xs[vs[:,i]>0],bins=bins,alpha=.5,log=True)
                ax[sidx].axvline(np.mean(xs[vs[:,i]>0]),color=colors[i],linewidth=5)
                ax[sidx].set_ylim(ymin=1,ymax=S)
                ax[sidx].set_xticks([0,3,6])
            sidx+=1
            #plt.subplot(3,3,3+sidx)
            
            #vmax=np.max(np.abs(np.dot(us,vs.T)/S)) *4
            #vmax=0.005
            
            #matplot(locmat_red)
        
        #plt.savefig('first.svg')

    table=pd.DataFrame(table).set_index('sigma')
    return table,mat


def do(us,vs):
    S,nf=us.shape
    fig, ax = plt.subplots(2,2,figsize=(10,10))

    table,mat=comp(us,vs,ax[0])
    
    mat=np.dot(us,vs.T)/S + mat*table.index[-1]/np.sqrt(S)

    ax[1,0].plot(table.index,table[[k for k in table if 'theo' in k]].values,linewidth=5)
    for k in table:
        if 'simu' in k:
            ax[1,0].scatter(table.index,table[k].values,s=140)
    
    ax[1,0].set_xticks([0,.5,1,1.5],labels =[0,.5,1,1.5], fontsize=24)
    ax[1,0].set_yticks([0,0.1,0.2,0.3,0.4],labels = [0,0.1,0.2,0.3,0.4],fontsize=24)
        
    
    colors=list(mcolors.TABLEAU_COLORS)
    
    for i in range(nf):
        ax[1,1].plot(table.index,[_[i] for _ in table['phi'].values],color=colors[i],linewidth=5)
        ax[1,1].scatter(table.index,[_[i] for _ in table['alive'].values],color=colors[i],s=140)
    
    ax[1,1].scatter(table.index,[_[-1] for _ in table['alive'].values],color='grey',s=140)
    ax[1,1].plot(table.index,[_[-1] for _ in table['phi'].values],color='grey',linewidth=5)
    
    ax[1,1].set_xticks([0,.5,1,1.5],labels = [0,.5,1,1.5],fontsize=24)
    ax[1,1].set_yticks([0,.5,1],labels = [0,.5,1],fontsize=24)
    #plt.yscale('log')
    
    plt.show()
    #plt.savefig('third.svg')
    return table

def main():

    S=1500
    nf=2

    vs=(np.arange(S)<S*.8).astype('float')
    vs=np.array([vs,1-vs]).T
    us=np.dot(vs,np.array([[0,-6],[-.5,0]]) )
    handled=False
    while not handled:
        try:
            tab = do(us,vs)
            handled=True
        except Exception as e:
            plt.close()
            if not 'DAMN' in str(e):
                handled = True


if __name__ == "__main__":
    main()