import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import math
import glob
import os


def read_arrs(pattern_files):
    files = glob.glob(pattern_files)
    files = sorted(files, reverse=True)

    arrs = []
    names = []

    for file in files:
        arr = np.loadtxt(file, delimiter=',')
        arrs.append(arr)

        _, name = os.path.split(file)
        names.append(name[:-4])


    return arrs, names


def generate_plots(curr_val, pattern, save_place):

    arrs, names = read_arrs(pattern)

    # Define subplots
    rows = int(math.ceil(len(arrs) / 3))
    cols = min(len(arrs), 3)
    fig, ax = plt.subplots(rows, cols, squeeze=False)

    k = 0
    for i in range(rows):
        if k >= len(arrs):
                break

        for j in range(cols):
            if k >= len(arrs):
                break

            arr = arrs[k]
            name = names[k]

            sns.distplot(arr, ax=ax[i][j])
            ymin, ymax = ax[i][j].get_ylim()
            ax[i][j].vlines(x=curr_val, ymin=ymin, ymax=ymax, color='b')
            ax[i][j].vlines(x=np.median(arr), ymin=ymin, ymax=ymax, color='r')
            print("Mediana para " + name + " " + str(np.median(arr)))
            ax[i][j].set_title(name)

            k += 1

    plt.show()
    # plt.savefig(save_place, dpi=150)


def main():
    pattern = 'outputs/preds/preds_CVS_*_201903*.csv'
    save_name = 'outputs/reports/pred_IPC.png'
    curr_val1 = 52.93

    generate_plots(curr_val1, pattern, save_name)

if __name__=='__main__':
    main()
