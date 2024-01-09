#%%
import matplotlib.pylab as plt
import numpy as np
import pwlf
# %%
exec(
    open('./calculator.log', 'r').read()
)
# %%
plt.plot(time, results[0])
plt.figure()
plt.plot(time, results[1], 'o-')
# %%
exp_data = np.genfromtxt(
    './model/indent_sph_ak4/sph.csv',
    skip_header=1,
    usecols=[0, 1, 2],
    unpack=True,
    delimiter=',',
)
# %%
plt.plot(exp_data[0], exp_data[2])
plt.plot(time, results[1], 'o-')
# %%
Fe = np.interp(time, exp_data[0], exp_data[2])
Fc = np.array(results[1])
plt.plot(Fe)
plt.plot(Fc)
# %%
mask1 = (Fc < -1000)
k = Fe/Fc
mask = mask1 & (k > 0)
plt.plot(Fe[mask]/Fc[mask])
k = Fe[mask]/Fc[mask]
# %%
ep = np.array(results[0])
sig = (lambda x: 200+1000*x)(ep)
plt.plot(results[0], sig)
plt.plot(np.array(results[0])[mask], sig[mask]*k)
pw = pwlf.PiecewiseLinFit(x=np.array(results[0])[mask], y=sig[mask]*k)
print(pw)
xx = np.linspace(0, 0.4, 50)
pw.fitfast(5, pop=2)
plt.plot(xx, pw.predict(xx), 's-')
plt.ylim(bottom=0)
# %%
plt.plot(exp_data[0], exp_data[2])
pp = pwlf.PiecewiseLinFit(exp_data[0], exp_data[2])
tt = np.linspace(0, 0.14, 20)
pp.fit_with_breaks(tt)
plt.plot(tt, pp.predict(tt))
# %%
mask1 = (Fc < -100)
k = Fe/Fc
mask = mask1 & (k > 0)
plt.plot(np.array(results[0])[mask], Fe[mask]/Fc[mask])
k = Fe[mask]/Fc[mask]
ep = np.array(results[0])
sig = (lambda x: 200+1000*x)(ep)
p = np.polyfit(np.array(results[0])[mask], Fe[mask]/Fc[mask], 2)
plt.plot(np.array(results[0]), np.polyval(p, np.array(results[0])))
# plt.plot(results[0], sig)
# plt.plot(np.array(results[0])[mask], sig[mask]*k)
plt.ylim(bottom=0)
# %%
plt.plot(results[0], sig)
plt.plot(results[0], sig*np.polyval(p, np.array(results[0])))

# %%
