import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

def print_confusion_matrix(confusion_matrix, class_names, figsize = (10,7), fontsize=14):
    df_cm = pd.DataFrame(
        confusion_matrix, index=class_names, columns=class_names, 
    )
    fig = plt.figure(figsize=figsize)
    try:
        heatmap = sns.heatmap(df_cm, annot=True, fmt="d")
    except ValueError:
        raise ValueError("Confusion matrix values must be integers.")
    heatmap.yaxis.set_ticklabels(heatmap.yaxis.get_ticklabels(), rotation=0, ha='right', fontsize=fontsize)
    heatmap.xaxis.set_ticklabels(heatmap.xaxis.get_ticklabels(), rotation=45, ha='right', fontsize=fontsize)
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
    return fig

def read(fname):
    with open(fname, 'r') as f:
        s = f.readlines()
    output, gt = [], []
    for l in s:
        t = l.split(' ')
        output.append(int(t[0]))
        gt.append(int(t[1][:-1]))
    return output, gt

if __name__ == '__main__':
    output, gt = read('output.txt')
    cm = confusion_matrix(gt, output)
    print_confusion_matrix(cm, ['forward', 'left', 'right', 'stop'])
    plt.savefig('confusion_matrix.png', dpi=300)