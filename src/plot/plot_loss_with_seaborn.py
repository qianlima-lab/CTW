import seaborn as sns
import matplotlib.pyplot as plt
sns.set_palette(['red','blue'])
#sns.set_style("whitegrid")
fig=plt.figure(dpi=200)
sns.set_style("darkgrid")
sns.distplot(standar_loss[labels[recreate_idx].detach().cpu().numpy()==7], hist = False, kde_kws = {'color':'r', 'linestyle':'-','shade':True},
             norm_hist = True,label = 'label 1')
sns.distplot(standar_loss[labels[recreate_idx].detach().cpu().numpy()==4], hist = False, kde_kws = {'color':'b', 'linestyle':'-','shade':True},
             norm_hist = True,label = 'label 2')
plt.legend(fontsize=26)
plt.tick_params(width=0.5, labelsize=10)
plt.xlabel('Loss',fontsize=20)
plt.ylabel('Density',fontsize=20)
plt.text(0.3,0.3,"Threadshold",fontdict={'size':'22','color':'k'})
plt.axvline(standar_loss.mean(),color='k',linestyle='--')
fig.tight_layout()
# plt.savefig('loss2.svg',format='svg')
plt.show()

import seaborn as sns
import matplotlib.pyplot as plt
sns.set_palette(['red','blue'])
#sns.set_style("whitegrid")
fig=plt.figure(dpi=200)
sns.set_style("darkgrid")
sns.distplot(loss_mean[labels.detach().cpu().numpy()==7], hist = False, kde_kws = {'color':'r', 'linestyle':'-','shade':True},
             norm_hist = True,label = 'label 1')
sns.distplot(loss_mean[labels.detach().cpu().numpy()==4], hist = False, kde_kws = {'color':'b', 'linestyle':'-','shade':True},
             norm_hist = True,label = 'label 2')
plt.legend(fontsize=26)
plt.tick_params(width=0.5, labelsize=10)
plt.xlabel('Loss',fontsize=20)
plt.ylabel('Density',fontsize=20)
fig.tight_layout()
# plt.savefig('loss1.svg',format='svg')

plt.show()
