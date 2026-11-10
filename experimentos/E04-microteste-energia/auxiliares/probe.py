import numpy as np, time, faiss
from zeus_apple_silicon import AppleEnergyMonitor
print('faiss',faiss.__version__, 'threads', faiss.omp_get_max_threads())
m=AppleEnergyMonitor()
m.begin_window('a'); t=time.time()
A=np.random.rand(2000,2000).astype('f4')
for _ in range(10): A@A
r=m.end_window('a'); print(time.time()-t); print(r)
