

import scipy.io as sio
import sys, os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from ptitprince import half_violinplot
import ptitprince as pt

# 创建一个包含两个子图的图形
fig, (ax1, ax2) = plt.subplots(1, 2)
ax1.set_title('(a)', x = 0.1)

ax2.set_title('(b)', x = 0.1)
ax1.set_position([0.1, 0.1, 0.3, 0.8])  # 设置第一个子图的位置和尺寸
ax2.set_position([0.45, 0.1, 0.5, 0.8])  # 设置第二个子图的位置和尺寸

# 加载数据
name = 'Degene_232'
data232 = sio.loadmat(os.path.join(sys.path[0], name + '.mat'))

data_overlap_232 = data232['m']

data_overlap_232 = {
    r'$|u_1 \rangle$': data_overlap_232[:,0],
    r'$|u_2 \rangle$': data_overlap_232[:,1]
}

colors_overlap_232 = ["#660874", "#AB2A3C"]
df_overlap_232 = pd.DataFrame(data_overlap_232)
print('overlap\n', df_overlap_232.describe())


# 创建 raincloud 图
pt.half_violinplot(data=df_overlap_232, inner=None, width=0.6, palette=colors_overlap_232, edgecolor="white", ax=ax1)
sns.stripplot(data=df_overlap_232, jitter=True, edgecolor="none", size=1.2, palette=colors_overlap_232, ax=ax1)
# sns.boxplot(data=df_overlap_232, width=0.15, zorder=10, whiskerprops={'linewidth': 1.5, "zorder": 10},
#             boxprops={'facecolor': 'none', "zorder": 10}, palette=colors_overlap_232, ax=ax1)

ax1.grid(True)
ax1.set_ylabel('Overlap')
ax1.set_ylim(-0.1, 1)

name = 'MG_re_plot'
data_MG = sio.loadmat(os.path.join(sys.path[0], 'Data', name + '.mat'))
data_MG = data_MG['m']

data_MG = {
    r'$|v_1 \rangle$': data_MG[:, 0],
    r'$|v_2 \rangle$': data_MG[:, 1],
    r'$|v_3 \rangle$': data_MG[:, 2],
    r'$|v_4 \rangle$': data_MG[:, 3],
    r'$|v_5 \rangle$': data_MG[:, 4],
}

colors_1 = ["#6FAE45", "#589CD6", "#4474C4", "#660874", "#AB2A3C"]

# 将数据转换为 DataFrame
df_MG = pd.DataFrame(data_MG)

# 使用 DataFrame 计算统计信息
stats = df_MG.describe()
print(stats)

# 创建 raincloud 图
ax2 = pt.half_violinplot(data=df_MG, inner=None, width=1, palette=colors_1, edgecolor="white")
ax2 = sns.stripplot(data=df_MG, jitter=True, edgecolor="none", size=1.2, palette=colors_1)
ax2.set_yticklabels([])
ax2.set_ylim(-0.1, 1)
# 添加网格线到图中
plt.grid(True)

# Adjust spacing between subplots
# plt.tight_layout()
# plt.show()

# save
path = os.path.join(sys.path[0],'Image','Rain_232_MG.pdf')
plt.savefig(path, format='pdf', bbox_inches='tight')
path = os.path.join(sys.path[0],'Image','Rain_232_MG.png')
plt.savefig(path, format='png', bbox_inches='tight')