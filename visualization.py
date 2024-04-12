import seaborn as sns
import matplotlib.pyplot as plt

import matplotlib
matplotlib.use('TkAgg')

def plot_spam_ham_count(df):
    plt.figure(figsize=(8, 6))  # Set the desired width and height of the plot

    # Create the countplot
    custom_palette = ["#1f77b4", "#ff7f0e"]
    ax = sns.countplot(data=df, x='Label', palette=custom_palette)

    # Add count labels on top of each bar
    for p in ax.patches:
        height = p.get_height()
        ax.text(p.get_x() + p.get_width() / 2, height, height, ha='center')

    plt.title('Number of Ham and Spam Emails')
    plt.xlabel('Label')
    plt.ylabel('Count')
    plt.xticks([0, 1], ['Ham', 'Spam'])

    plt.show()
