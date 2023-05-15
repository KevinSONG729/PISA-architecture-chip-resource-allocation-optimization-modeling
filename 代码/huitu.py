import numpy as np
import csv
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt

data = np.loadtxt(open("KedaMatrix.csv","rb"),delimiter=",",skiprows=0)
print(data)
sns.set()
plt.rcParams['font.sans-serif']='SimHei'
ax = sns.heatmap(data, cmap="Purples")
plt.title('基本块间可达分布', fontsize=10, fontweight="bold")
plt.xticks(fontsize=10)
plt.yticks(fontsize=10)
plt.savefig('C:\\Users\\Desktop\\研赛\\D\\heatmap2.png',dpi=1200, bbox_inches='tight')
plt.show()    