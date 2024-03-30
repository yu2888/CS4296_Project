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

    plt.title('Number of ham and spam emails')
    plt.show()

def confusion_matrix_plot(matrix, model_name):
    # Plot the confusion matrix using a heatmap
    plt.figure(figsize=(5,5))
    ax= plt.subplot()
    sns.set(font_scale=1)
    sns.heatmap(matrix, annot=True, ax=ax, cmap='Blues')

    # Set the axis labels, title, and tick labels for the plot
    ax.set_xlabel('Predicted label', size=10)
    ax.set_ylabel('True label', size=10)
    ax.set_title(model_name, size=15) 
    ax.xaxis.set_ticklabels(["ham","spam"], size=15)
    ax.yaxis.set_ticklabels(["ham","spam"], size=15)
    plt.show()
