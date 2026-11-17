"""Confere rho_prev contra o modelo de brinquedo em 12 combinações de N, r e SNR_B."""
import warnings; warnings.filterwarnings("ignore")
import numpy as np
from toy import run
for N in [1000,4000,16000]:
    for r in [4,16]:
        for sb in [0.1,1.0]:
            o=run(N=N,r=r,snr_b=sb,T=60000)
            snr_ef=1/(1/sb+(1+1/sb)/15)
            pred=np.sqrt(2/np.pi)*np.sqrt(N/r)*np.sqrt(1+1/snr_ef)
            print(N,r,sb,'T',o['T'],'rho',round(o['M']/o['Dmin'],1),'pred',round(pred,1),'util/Dmin',round(o['useful']/o['Dmin'],2))
