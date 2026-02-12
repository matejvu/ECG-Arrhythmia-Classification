import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.feature_selection import SelectKBest, f_classif


if __name__ == "__main__":
    # Load your data
    df = pd.read_csv("./data_arrhythmia.csv", na_values=['?'])


    # Missing data
    missing_pct = df.isna().sum() / len(df) * 100
    print(missing_pct.sort_values(ascending=False))

    # Remove column J because it has a high percentage of missing values
    df = df.drop(columns=["J", "P"])
    #Impute column T because missing values
    df['T'] = df['T'].fillna(df['T'].median())
    #Remove the rows with mising QRST and heart_rate
    df = df.dropna(subset=["QRST", "heart_rate"])


    labels = df["diagnosis"].values
    features = df.drop(columns=["diagnosis"]).values
    
    # Plot the distribution of diagnosis labels
    counts = df["diagnosis"].value_counts()
    counts.plot(kind='bar', title='Distribution of Diagnosis Labels')
    plt.xlabel('Diagnosis')
    plt.ylabel('Number of Instances')
    plt.show()

    # Distribution of binary classes (Abnormal - 1, Normal - 0)
    labels_binary = np.where(labels > 1, 1, 0)
    print("Distribution of binary labels:")
    print(pd.Series(labels_binary).value_counts())

    #
    best_features = SelectKBest(score_func=f_classif, k=5)
    fit = best_features.fit(features, labels_binary)
    feature_scores = pd.DataFrame({'Feature': labels[:-1], 'Score': fit.scores_}).sort_values(by='Score', ascending=False)
    print(feature_scores)