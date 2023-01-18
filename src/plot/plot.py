import os.path
import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import pyplot
plt.style.use('seaborn-whitegrid')
palette = pyplot.get_cmap('Set1')


plt.rcParams.update({'font.size':22})
confnum = pd.read_csv(os.path.join(
    os.getcwd(), 'ECG5000Sym30_minmax.csv'))
#

sns.set_style('whitegrid')
# for epoch in [50,100,150,200,250,290]:
acc_dict={'min-max':[],'z-score':[]}
for seed in [37, 118, 337, 815, 19]:
    newdf=confnum.groupby(['epoch', 'method', 'seed'], as_index=False).sum()
    for mt in ['min-max','z-score']:
        acc_seed_method = newdf[newdf.seed==seed][newdf.method==mt]['TP']/newdf[newdf.seed==seed][newdf.method==mt]['total']
        acc_dict[mt].append(np.array(acc_seed_method))

fig=plt.figure(figsize=(20,10))
iters=list(range(len(acc_dict['min-max'][0])))
length=len(iters)
color = palette(2)
ax = fig.add_subplot(1, 1, 1)
avg = np.mean(acc_dict['min-max'], axis=0)[:length]
std = np.std(acc_dict['min-max'], axis=0)[:length]
r1 = list(map(lambda x: x[0] - x[1], zip(avg, std)))
r2 = list(map(lambda x: x[0] + x[1], zip(avg, std)))
ax.plot(iters, avg, color=color, label="min-max", linewidth=3.0)
ax.fill_between(iters, r1, r2, color=color, alpha=0.2)

color = palette(0)
avg = np.mean(acc_dict['z-score'], axis=0)[:length]
std = np.std(acc_dict['z-score'], axis=0)[:length]
r1 = list(map(lambda x: x[0] - x[1], zip(avg, std)))
r2 = list(map(lambda x: x[0] + x[1], zip(avg, std)))
ax.plot(iters, avg, color=color, label="z-score", linewidth=3.0)
ax.fill_between(iters, r1, r2, color=color, alpha=0.2)


font1 = {'weight': 'normal', 'size': 25}
ax.legend(loc='best', prop=font1)
ax.set_xlabel('epochs', fontsize=25)
ax.set_ylabel('Precision', fontsize=25)

plt.show()
