import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


class Vizualizer:
    def plot_wine_data(self, red, white):
        fig, ax = plt.subplots(1, 2)

        ax[0].hist(red.alcohol, 10, facecolor='red', alpha=0.5, label="Red wine")
        ax[1].hist(white.alcohol, 10, facecolor='white', ec="black", lw=0.5, alpha=0.5, label="White wine")

        # fig.subplots_adjust(left=0, right=1, bottom=0, top=0.5, hspace=0.05, wspace=1)
        ax[0].set_ylim([0, 1000])
        ax[0].set_xlabel("Alcohol in % Vol")
        ax[0].set_ylabel("Frequency")
        ax[1].set_xlabel("Alcohol in % Vol")
        ax[1].set_ylabel("Frequency")
        # ax[0].legend(loc='best')
        # ax[1].legend(loc='best')
        fig.suptitle("Distribution of Alcohol in % Vol")

        plt.show()

    def plot_quality_from_sulphates(self, red, white):
        fig, ax = plt.subplots(1, 2, figsize=(8, 4))

        ax[0].scatter(red['quality'], red["sulphates"], color="red")
        ax[1].scatter(white['quality'], white['sulphates'], color="white", edgecolors="black", lw=0.5)

        ax[0].set_title("Red Wine")
        ax[1].set_title("White Wine")
        ax[0].set_xlabel("Quality")
        ax[1].set_xlabel("Quality")
        ax[0].set_ylabel("Sulphates")
        ax[1].set_ylabel("Sulphates")
        ax[0].set_xlim([0, 10])
        ax[1].set_xlim([0, 10])
        ax[0].set_ylim([0, 2.5])
        ax[1].set_ylim([0, 2.5])
        fig.subplots_adjust(wspace=0.5)
        fig.suptitle("Wine Quality by Amount of Sulphates")

        plt.show()

    def plot_quality_from_volatile_acidity(self, red, white):
        np.random.seed(570)

        redlabels = np.unique(red['quality'])
        whitelabels = np.unique(white['quality'])

        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(1, 2, figsize=(8, 4))
        redcolors = np.random.rand(6, 4)
        whitecolors = np.append(redcolors, np.random.rand(1, 4), axis=0)

        for i in range(len(redcolors)):
            redy = red['alcohol'][red.quality == redlabels[i]]
            redx = red['volatile acidity'][red.quality == redlabels[i]]
            ax[0].scatter(redx, redy, c=redcolors[i])
        for i in range(len(whitecolors)):
            whitey = white['alcohol'][white.quality == whitelabels[i]]
            whitex = white['volatile acidity'][white.quality == whitelabels[i]]
            ax[1].scatter(whitex, whitey, c=whitecolors[i])

        ax[0].set_title("Red Wine")
        ax[1].set_title("White Wine")
        ax[0].set_xlim([0, 1.7])
        ax[1].set_xlim([0, 1.7])
        ax[0].set_ylim([5, 15.5])
        ax[1].set_ylim([5, 15.5])
        ax[0].set_xlabel("Volatile Acidity")
        ax[0].set_ylabel("Alcohol")
        ax[1].set_xlabel("Volatile Acidity")
        ax[1].set_ylabel("Alcohol")
        # ax[0].legend(redlabels, loc='best', bbox_to_anchor=(1.3, 1))
        ax[1].legend(whitelabels, loc='best', bbox_to_anchor=(1.3, 1))
        # fig.suptitle("Alcohol - Volatile Acidity")
        fig.subplots_adjust(top=0.85, wspace=0.7)

        plt.show()

    def plot_corelation_matrix(self, wines):
        corr = wines.corr()
        sns.set()
        sns.heatmap(corr,
                    xticklabels=corr.columns.values,
                    yticklabels=corr.columns.values)
        # # TODO use seaborn's palette
        plt.show()

    def plot_distribution(self, x):
        sns.distplot(x)

