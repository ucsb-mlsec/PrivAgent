import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

# Read data from CSV
df = pd.read_csv('sampling.csv')
# remove the last few rows
df = df[:-20]
# Create shorter names for better legend display
column1 = 'llama3.1_8B_rl_finetune_bonus_sampling - env/reward_mean'
column2 = 'llama3.1_8B_rl_finetune_bonus - env/reward_mean'
display_name1 = 'With Dynamic Temperature'
display_name2 = 'W/O Dynamic Temperature'

plt.figure(figsize = (12, 6))

# Plot main curves
sns.lineplot(data = df, x = df.index, y = column1,
             linewidth = 2, label = display_name1, color = 'blue')
sns.lineplot(data = df, x = df.index, y = column2,
             linewidth = 2, label = display_name2, color = 'red')

# Add smoothed curves
window_size = 5
smoothed1 = df[column1].rolling(window = window_size, center = True).mean()
smoothed2 = df[column2].rolling(window = window_size, center = True).mean()

plt.plot(df.index, smoothed1, '--', color = 'lightblue', alpha = 0.7,
         linewidth = 2)
plt.plot(df.index, smoothed2, '--', color = 'lightcoral', alpha = 0.7,
         linewidth = 2)

# Customize the plot
# set fontsize for x and y ticks
plt.xticks(fontsize = 18)
plt.yticks(fontsize = 18)
plt.xlabel('Training Steps', fontsize = 24)
plt.ylabel('Average Reward', fontsize = 24)
plt.grid(True, linestyle = '--', alpha = 0.5)

# Adjust legend
plt.legend(loc = 'lower right', fontsize = 20)

# Optimize layout
plt.tight_layout()
# save to pdf
plt.savefig('sampling.pdf', transparent = True)
# Show plot
plt.show()
