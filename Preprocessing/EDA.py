import matplotlib.pyplot as plt
import seaborn as sns

def plot_distribution(df, column, title, filename):
    """Plot and save distribution of a column."""
    plt.figure(figsize=(10, 6))
    sns.countplot(data=df, x=column, order=df[column].value_counts().index)
    plt.title(title)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.show()
