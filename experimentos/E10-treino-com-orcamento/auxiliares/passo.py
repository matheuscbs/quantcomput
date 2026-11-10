import time, torch, torch.nn.functional as F, importlib.util, sys
sp=importlib.util.spec_from_file_location('t',str(__import__("pathlib").Path(__file__).resolve().parents[1] / "treino_com_orcamento.py")); t=importlib.util.module_from_spec(sp); sp.loader.exec_module(t)
for D_b,N in ((192,512),(128,512),(128,256)):
    t.D=D_b; t.NMAX=N
    m=t.Modelo(90).to(t.DEV); o=torch.optim.AdamW(m.parameters())
    x=torch.randint(0,90,(32,N+1)).to(t.DEV)
    for i in range(35):
        if i==5: t0=time.time()
        l=F.cross_entropy(m(x[:,:-1],16).reshape(-1,90),x[:,1:].reshape(-1)); o.zero_grad(); l.backward(); o.step(); float(l)
    print(D_b,N,round((time.time()-t0)/30*1000),'ms/passo',flush=True)
