"""Modelo quadrático de brinquedo: treino projetado ruidoso, cache sináptico e cache com decaimento (pré-registro do E17)."""
import numpy as np
def run(N=4000,k=20,r=8,lr=None,tau=None,kap=None,seed=0,T=20000,snr_b=0.3):
    rng=np.random.default_rng(seed)
    U=np.linalg.qr(rng.standard_normal((N,k)))[0]; h=np.linspace(1,0.2,k)
    wstar=U@rng.standard_normal(k)*3
    loss=lambda w: 0.5*np.sum(h*(U.T@(w-wstar))**2)
    w0=np.zeros(N); wd=w0.copy(); u=np.zeros(N); L0=loss(w0)
    lr=lr or 0.5*N/r
    M=0;Mc=0;load=0;sig=None
    for t in range(1,T+1):
        w=wd+u
        G=U@(h*(U.T@(w-wstar)))
        g=G+np.linalg.norm(G)/np.sqrt(N*snr_b)*rng.standard_normal(N)
        V=rng.standard_normal((N,r));V/=np.linalg.norm(V,axis=0)
        p=V.T@g; p+=np.sqrt(np.mean(p**2)/15)*rng.standard_normal(r)
        d=-lr/N*(V@p)*1.0
        M+=np.abs(d).sum()
        if sig is None: sig=np.sqrt(np.mean(d**2))
        if tau: u*=(1-1/tau)
        u+=d
        if kap is None: wd+=u; Mc+=np.abs(u).sum(); u[:]=0
        else:
            m=np.abs(u)>=kap*sig; Mc+=np.abs(u[m]).sum(); wd[m]+=u[m]; u[m]=0; load+=np.abs(u).sum()
        if loss(wd+u)<=0.1*L0: break
    useful=np.abs(U@(U.T@(wd+u-w0))).sum(); Dmin=np.abs(wd+u-w0).sum()
    return dict(T=t,M=round(M),Mc=round(Mc+np.abs(u).sum()),Dmin=round(Dmin),useful=round(useful),load=round(load/t))
if __name__ == "__main__":
    print('sem cache   ',run())
    for kap in [4,16,64]: print('cache k',kap,run(kap=kap))
    for tau in [10,30,100,300]:
        for kap in [4,16]: print('decai tau',tau,'k',kap,run(kap=kap,tau=tau))
