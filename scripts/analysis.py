from google.colab import files
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

#df = pd.read_csv('base.csv')
uploaded = files.upload()
df = pd.read_csv(list(uploaded.keys())[0])

df_clean = df.dropna(subset=['Critic_Score', 'Critic_Count', 'Genre'])

sns.set_theme(style="ticks")
fig, axes = plt.subplots(1, 3, figsize=(16, 5))

# гистограмма распределения Critic_Score
sns.histplot(df_clean['Critic_Score'], bins=30, color='#4c72b0', ax=axes[0])
axes[0].set_title('Распределение Critic_Score', fontsize=10)
axes[0].set_xlabel('Оценка критиков', fontsize=9)
axes[0].set_ylabel('Число игр', fontsize=9)
axes[0].tick_params(axis='both', labelsize=8)

# боксплот Critic_Score по жанрам
sns.boxplot(data=df_clean, x='Genre', y='Critic_Score', ax=axes[1], 
            color='white', fliersize=3, linewidth=1)
axes[1].set_title('Critic_Score по жанрам', fontsize=10)
axes[1].set_xlabel('')
axes[1].set_ylabel('')
axes[1].tick_params(axis='x', rotation=90, labelsize=8) # поворот подписей жанров
axes[1].tick_params(axis='y', labelsize=8)

# Сскеттерплот Critic_Count vs Critic_Score
sns.scatterplot(data=df_clean, x='Critic_Count', y='Critic_Score', 
                color='#55a868', alpha=0.4, s=15, edgecolor=None, ax=axes[2])
axes[2].set_title('Critic_Count vs Critic_Score', fontsize=10)
axes[2].set_xlabel('Число критиков, поставивших оценку', fontsize=9)
axes[2].set_ylabel('Оценка критиков', fontsize=9)
axes[2].tick_params(axis='both', labelsize=8)

# автоматическая настройка отступов и отображение
plt.tight_layout()
plt.show()