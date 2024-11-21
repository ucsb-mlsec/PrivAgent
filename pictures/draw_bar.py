import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize = (15, 6))

data1 = pd.DataFrame({
    'System': ['WES', 'ROUGE', 'SS'],
    'Score': [0.718, 0.45, 0.12]
})
# System is the index
# data1 = data1.set_index('System', inplace = True)

data2 = pd.DataFrame({
    'System': ['PrivAgent', 'FixedT'],
    'Score': [0.718, 0.707]
})

data3 = pd.DataFrame({
    'System': ['PrivAgent', 'NoDiv'],
    'Score': [0.718, 0.64]
})

colors = ['#1F77B4', '#B30000', '#2E7D32']
sns.barplot(data = data1,
            x = 'System',
            y = 'Score',
            ax = ax1,
            palette = colors)
ax1.grid(True, axis = 'y', linestyle = '--', alpha = 0.3)
# ax1.set_xticklabels(data1["System"], fontsize=14)
ax1.set_title('Reward Functions', fontsize = 24)
# set font size for x and y ticks
ax1.set_xticklabels(data1["System"], fontsize=22)
ax1.set_ylabel('Attack Performance', fontsize = 24)
ax1.set_yticklabels([f'{y:.2f}' for y in ax1.get_yticks()], fontsize=18)
# remove xticks
ax1.set_xlabel("")
# customize_axis(ax1, data1['System'].tolist())
# ax1.grid(True, axis = 'y', linestyle = '--', alpha = 0.3)
# ax1.set_ylabel('Attack Performance', fontsize = 18)

sns.barplot(data = data2, x = 'System', y = 'Score', ax = ax2, palette = colors[:2],width = 0.6)
ax2.set_title('Temp Adjustment', fontsize = 24)
ax2.grid(True, axis = 'y', linestyle = '--', alpha = 0.3)
ax2.set_ylabel('')
# set font size for x and y ticks
ax2.set_xticklabels(data2["System"], fontsize=22)
ax2.set_xlabel("")
ax2.set_yticklabels([f'{y:.2f}' for y in ax2.get_yticks()], fontsize=18)
# Diversity
sns.barplot(data = data3, x = 'System', y = 'Score', ax = ax3, palette = colors[:2],width = 0.6)
ax3.set_title('Diversity Reward', fontsize = 24)
ax3.grid(True, axis = 'y', linestyle = '--', alpha = 0.3)
ax3.set_ylabel('')
ax3.set_ylim(0.61, 0.725)
# set font size for x and y ticks
ax3.set_xticklabels(data3["System"], fontsize=22)
ax3.set_xlabel("")
plt.yticks(fontsize = 18)

plt.tight_layout()

plt.savefig('ablation_study.pdf', dpi = 300, bbox_inches = 'tight', transparent = True)
plt.show()
