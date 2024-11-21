import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

# Read data from CSV
df = pd.read_csv('train_data.csv')

sns.set_palette("husl")

# Create figure
plt.figure(figsize = (12, 6))

df['reward_mean'] = df['training_data_sliding_stage2 - env/reward_mean']
# Plot main curve
sns.lineplot(data = df, x = 'Step', y = 'reward_mean',
             linewidth = 2.5, label = 'Original')

# Add smoothed curve
window_size = 5
smoothed_rewards = df['training_data_sliding_stage2 - env/reward_mean'].rolling(window = window_size,
                                                                                center = True).mean()
plt.plot(df['Step'], smoothed_rewards, '--', color = 'red', alpha = 0.7, linewidth = 2)

# Set title and labels
plt.xlabel('Training Steps', fontsize = 24)
plt.ylabel('Average Reward', fontsize = 24)

plt.xticks(fontsize = 18)
plt.yticks(fontsize = 18)
# Add grid
plt.grid(True, linestyle = '--', alpha = 0.7)

# Adjust legend and larger font size
plt.legend(fontsize = 18)

# Optimize layout
plt.tight_layout()

# save to pdf
plt.savefig('train_curve.pdf', transparent = True)
# Show plot
plt.show()