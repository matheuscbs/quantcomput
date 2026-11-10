import time, torch, torch.nn.functional as F, importlib.util
torch.set_num_threads(8)
sp=importlib.util.spec_from_file_location('t',str(__import__("pathlib").Path(__file__).resolve().parents[1] / "treino_com_orcamento.py")); t=importlib.util.module_from_spec(sp); sp.loader.exec_module(t)
t.D=128;t.NMAX=256;t.BATCH=32
for dev,bud in (('mps',None),('mps',16),('cpu',None),('cpu',16)):
    t.DEV=dev
    m=t.Modelo(90).to(dev); o=torch.optim.AdamW(m.parameters())
    x=torch.randint(0,90,(32,257)).to(dev)
    for i in range(15):
        if i==5: t0=time.time()
        l=F.cross_entropy(m(x[:,:-1],bud).reshape(-1,90),x[:,1:].reshape(-1)); o.zero_grad(); l.backward(); o.step(); l.item()
    print(dev,bud,round((time.time()-t0)/10*1000),'ms/passo',flush=True)
