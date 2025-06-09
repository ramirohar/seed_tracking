#%%
import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
from movement_tracking import select_ROI
#%%
data = pd.read_pickle("tracking_results_2//DELTA45_TIMESTEP100_NSTEPS70_REGLA//trajectory.pickle")
#%%
plt.plot(data.x[0:600])
trim = data.x[270:490]
plt.plot(data.x[270:490])
#%%
pos = trim.x
t = np.arange(len(pos)) * 1/data.attrs["fps"]

plt.plot(t,pos)

#%%
m,b  = np.polyfit(x=t,y=pos, deg=1)

plt.plot(t,pos)
adj = t*m+b
plt.plot(t, adj)
# %%
diff = pos - adj
plt.plot(t, diff)
#%%

d = np.arange(256)
sp = np.fft.fft(np.sin(2*d))
freq = np.fft.fftfreq(d.shape[-1])
plt.plot(freq, sp.real, freq, sp.imag)

w = freq[np.where(sp.imag == max(sp.imag))[0]]
a = 1/(2*np.pi)
w/a
#%%
n=t.shape[-1]

sp = np.fft.fft(diff)
freq = np.fft.fftfreq(n, d=1/data.attrs["fps"])[:n//2]
abs_sp = abs(sp[:n//2])
plt.plot(freq, abs_sp)

sp[np.where(abs(sp) != max(abs(sp)))]  =  0 

reconstructed_signal = np.fft.ifft(sp)

dominant_freq = freq[np.argmax(abs_sp)]
dominant_freq
#%%
off = 0.5
plt.plot(t,diff, label= "original")
# plt.plot(t, np.cos(2*np.pi * dominant_freq * (t+off)))
plt.plot(t, reconstructed_signal, label=f"reconstructed f = {dominant_freq:.3} Hz")
plt.xlabel("time [s]")
plt.ylabel("diff [px]")
plt.legend()
#%%


def force_model_1(x,v):
    F0 = 1
    A = 50
    w = 0.5
    Fd = 0.1
    return F0 + A*np.sin(w*x) - v*Fd

def force_model(x,v):
    F0 = 1
    A = 1
    return F0*(x)**2 + A*np.sin(w*x)

def loop(x_0, v_0,force, dt = 0.01):
    x = x_0
    v = v_0

    xs = []
    ts = []

    for i in range(1000):    
        
        a = force(x,v)
        x += v*dt
        v += a*dt    

        xs.append(x)
        ts.append(i*dt)

    return xs, ts

xs,ts = loop(0,0.5,force_model)
plt.plot(ts,xs)

# %%
