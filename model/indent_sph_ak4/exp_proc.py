#%%
import numpy as np
import matplotlib.pylab as plt
import os
# %%
def readOSC(file_path: str)->None|list:
    osc_file = file_path
    if os.path.exists(osc_file):
        print('Reading .osc')
        k = 0.038
        data = np.load(osc_file)
        dt = data['data']['dt'] * 1e3
        dy = data['data']['dy'] * k
        y0 = data['data']['y0']
        y = data['data']['data']
        yy = []
        for i in range(2):
            yy.append(y[i] * dy[i] + y0[i])
            yy[-1] -= yy[-1][:100].mean()
        return [dt[0] * np.arange(len(y[0])), *yy]
    else:
        return None
def movingAverage(data, degree=10):
    smoothed = np.zeros(len(data))  # - degree + 1)
    for i in range(degree // 2, len(smoothed) - degree // 2):
        smoothed[i] = sum(data[i:i + degree]) / degree
    return smoothed
# %%
if (rez:=readOSC('./sph-06.osc')) is not None:
    t, e1, e2 = rez
    e1 = movingAverage(e1, 20)    
    e2 = movingAverage(e2, 20)    
    plt.plot(e1)
    plt.plot(e2)
    plt.figure()
    DELTA = 0
    N1 = 520+DELTA
    N2 = 1390+DELTA
    N3 = 1380+DELTA
    DN = 580-DELTA
    ei = -e1[N1:N1+DN]
    er =  e1[N2:N2+DN]
    et = -e2[N3:N3+DN]
    t = t[:DN]
    plt.plot(ei)
    plt.plot(er)
    plt.plot(et)
    plt.plot(ei-er, 'k--')
    plt.axhline(20/2/4850, color='k')
# %%
Eb = 185000
cb = 4850
rb = 10
V = cb*(ei+er-et)
F = Eb*np.pi*rb**2*et
plt.plot(t, V)
plt.figure()
plt.plot(t, F)
with open('sph.csv', 'w') as fout:
    fout.write('t, v, f\n')
    for i in range(DN):
        fout.write(f'{t[i]}, {V[i]}, {-F[i]/2/np.pi}\n')
# %%
