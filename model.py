import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.discriminant_analysis import StandardScaler
from sklearn.feature_selection import SelectKBest, VarianceThreshold, f_classif, mutual_info_classif
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix
from sklearn.model_selection import cross_val_score
from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import make_pipeline


def remove_constant_features(df, target_column="diagnosis", threshold=0):    
    labels = df[target_column].values
    features_df = df.drop(columns=[target_column])
    
    selector = VarianceThreshold(threshold=threshold)
    features_transformed = selector.fit_transform(features_df)
    
    # Zadrži samo nekonstantne kolone
    features_clean_df = pd.DataFrame(
        features_transformed,
        columns=features_df.columns[selector.get_support()],
        index=features_df.index
    )
    
    features_clean_df[target_column] = labels
    
    return features_clean_df

def plot_cf_matrix(X, y, clf, class_names=None):
    y_pred = clf.predict(X)
    disp = ConfusionMatrixDisplay(
        confusion_matrix=confusion_matrix(y, y_pred),
        display_labels=class_names
    )

    disp.plot(cmap='Blues', values_format='d')
    plt.title('Confusion Matrix')
    plt.show()

def handle_missing_data(df, verbose = False):
    # Missing data
    missing_pct = df.isna().sum() / len(df) * 100
    if verbose:
        print('Missing data statistics:')
        print(missing_pct.sort_values(ascending=False))

    # Remove column J because it has a high percentage of missing values
    df = df.drop(columns=["J", "P"])
    #Impute column T because missing values
    df['T'] = df['T'].fillna(df['T'].median())
    #Remove the rows with mising QRST and heart_rate
    df = df.dropna(subset=["QRST", "heart_rate"])
    return df

def select_top_features(features, labels, df, heuristic="ANOVA", k=-1, verbose=False):
    # Select scoring function
    if heuristic == "ANOVA":
        best_features = SelectKBest(score_func=f_classif, k=k)
    elif heuristic == "MutualInfo":
        best_features = SelectKBest(score_func=mutual_info_classif, k=k)
    else:
        raise ValueError("Unknown heuristic")
    
    fit = best_features.fit(features, labels)
    feature_names = df.drop(columns=["diagnosis"]).columns

    feature_scores = (
        pd.DataFrame({
            "Feature": feature_names,
            "Score": fit.scores_
        })
        .sort_values(by="Score", ascending=False)
        .reset_index(drop=True)
    )

    if verbose:
        print(f"Feature scores based on {heuristic}:")
        print(feature_scores.head(k if k > 0 else len(feature_scores)))

    mask = best_features.get_support()
    selected_features = feature_names[mask]
    df_selected = df[selected_features.tolist() + ["diagnosis"]]

    features_selected = features[:, mask]

    return df_selected, features_selected

def test_model_performance(features, labels, clf, test_size=0.2):

    indices = np.random.permutation(len(features))
    features_shuffled = features[indices]
    labels_shuffled = labels[indices]

    # split index
    split = int((1 - test_size) * len(features))

    train_features = features_shuffled[:split]
    test_features  = features_shuffled[split:]
    train_labels   = labels_shuffled[:split]
    test_labels    = labels_shuffled[split:]

    # train + evaluate
    model = clf.fit(train_features, train_labels)
    accuracy = model.score(test_features, test_labels)

    print(f"Test accuracy: {accuracy:.4f}")

    return accuracy


if __name__ == "__main__":

    np.random.seed(24)
    # Load your data
    df = pd.read_csv("./data_arrhythmia.csv", na_values=['?'])

    df = handle_missing_data(df, verbose=False)
    df = remove_constant_features(df, "diagnosis", threshold=0)

    labels = df["diagnosis"].values
    features = df.drop(columns=["diagnosis"]).values
    

    # Plot the distribution of diagnosis labels
    # counts = df["diagnosis"].value_counts()
    # counts.plot(kind='bar', title='Distribution of Diagnosis Labels')
    # plt.xlabel('Diagnosis')
    # plt.ylabel('Number of Instances')
    # plt.show()

    # Distribution of binary classes (Abnormal - 1, Normal - 0)
    labels_binary = np.where(labels > 1, 1, 0)
    print("Distribution of binary labels:")
    print(pd.Series(labels_binary).value_counts())

    #BINARY PROBLEM
    
    

    df2, features_selected = select_top_features(features, labels_binary, df, heuristic="ANOVA", k=200, verbose=False)

    #Logistic Regression----------------------------------------------------------
    results = []
    #Cs = np.linspace(0.01, 0.035, 50)#[0.1, 0.5, 1.0, 2, 5, 10.0]
    Copt = 0.015

    clf = LogisticRegression(random_state=42,
                                penalty='l2',
                                C=Copt)
    pipe = make_pipeline(StandardScaler(), clf)
    # results.append(cross_val_score(pipe, features_selected, labels_binary, cv=5))

    # results = [res.mean() for res in results]
    # plt.figure()
    # plt.plot(Cs, results)
    # plt.show()
    # print("Cross-validation scores:")
    # print(results[0].mean())

    model = pipe.fit(features_selected, labels_binary)
    # plot_cf_matrix(features_selected, labels_binary, model, class_names=["Normal", "Arrhythmia"])
    test_model_performance(features_selected, labels_binary, pipe)

    #Naive Bayes------------------------------------------------------------------

    Ks = range(1,50,1)#[1,2,3,5, 10, 20, 50, 100, 200 ]
    Kopt = 11
    results = []
    #for k in Ks:
    df2, features_selected = select_top_features(features, labels_binary, df, heuristic="ANOVA", k=Kopt, verbose=False)
    clf = GaussianNB()
    # results.append(cross_val_score(clf, features_selected, labels_binary, cv=5))

    # results = [res.mean() for res in results]
    # plt.figure()
    # plt.plot(Ks, results)
    # plt.show()
    # print("Cross-validation scores:")
    # print(results[0].mean())

    model = clf.fit(features_selected, labels_binary)
    # plot_cf_matrix(features_selected, labels_binary, model, class_names=["Normal", "Arrhythmia"])
    test_model_performance(features_selected, labels_binary, clf)
    #Support Vector Machine-------------------------------------------------------




    #MULTICLASS PROBLEM


    # best_features = SelectKBest(score_func=mutual_info_classif, k=10)
    # fit = best_features.fit(features, labels_binary)
    # feature_scores = pd.DataFrame({'Feature': df.columns[:-1], 'Score': fit.scores_}).sort_values(by='Score', ascending=False)
    # print("Feature scores based on Mutual info:")
    # print(feature_scores[:10])
