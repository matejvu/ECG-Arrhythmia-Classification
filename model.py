import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.discriminant_analysis import StandardScaler
from sklearn.feature_selection import SelectKBest, VarianceThreshold, f_classif, mutual_info_classif
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix, precision_score, recall_score, accuracy_score
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import make_pipeline
from sklearn.svm import SVC


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

def test_model_performance(features, labels, clf, test_size=0.2, random_state=42):

    X_train, X_test, y_train, y_test = train_test_split(
        features, labels,
        test_size=test_size,
        shuffle=True,
        random_state=random_state
    )

    model = clf.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = 2 * (precision * recall) / (precision + recall + 1e-10)

    print(f"//=====TEST SCORES====\\\\")
    print(f"|| Accuracy : {acc:.4f}  ||")
    print(f"|| Precision: {precision:.4f}  ||")
    print(f"|| Recall   : {recall:.4f}  ||")
    print(f"|| F1 Score : {f1:.4f}  ||")
    print(f"\\\\====================//")

    return acc, precision, recall, f1


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

    # model = pipe.fit(features_selected, labels_binary)
    # plot_cf_matrix(features_selected, labels_binary, model, class_names=["Normal", "Arrhythmia"])
    # test_model_performance(features_selected, labels_binary, pipe)

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

    # model = clf.fit(features_selected, labels_binary)
    # plot_cf_matrix(features_selected, labels_binary, model, class_names=["Normal", "Arrhythmia"])
    # test_model_performance(features_selected, labels_binary, clf)
    
    #Support Vector Machine-------------------------------------------------------
    results = []
    kernels = 'rbf', 'poly', 'sigmoid'
    kernel_opt = 'rbf'
    Cs = np.linspace(0.1, 6, 30)#[0.01, 0.1, 1,3,5, 10]
    Copt = 0.9
    Ks = range(50, 120,2)#[1,2,3,5, 10, 20, 50, 100, 200 ]
    Kopt = 100
    
    df2, features_selected = select_top_features(features, labels_binary, df, heuristic="ANOVA", k=Kopt, verbose=False)

    # for kernel in kernels:
    # for k in Ks:
    #     df2, features_selected = select_top_features(features, labels_binary, df, heuristic="ANOVA", k=k, verbose=False)
    #     r = []
    #     # for C in Cs:
    clf = SVC(kernel=kernel_opt, C=Copt)
    pipe = make_pipeline(StandardScaler(), clf)
    #     r.append(cross_val_score(pipe, features_selected, labels_binary, cv=5))
    #     results.append(r[0])

    # results = [res.mean() for res in results]
    # plt.figure()
    # for ind, r in enumerate(results):
        # r = [r.mean() for r in r]
        # plt.plot(Cs, r, label=f'kernel: {kernels[ind]}')
    # plt.plot(Ks, results)
    # plt.legend()
    # plt.show()
    # print("Cross-validation scores:")
    # print(results[0].mean())

    model = pipe.fit(features_selected, labels_binary)
    plot_cf_matrix(features_selected, labels_binary, model, class_names=["Normal", "Arrhythmia"])
    test_model_performance(features_selected, labels_binary, pipe)



    #MULTICLASS PROBLEM


    # best_features = SelectKBest(score_func=mutual_info_classif, k=10)
    # fit = best_features.fit(features, labels_binary)
    # feature_scores = pd.DataFrame({'Feature': df.columns[:-1], 'Score': fit.scores_}).sort_values(by='Score', ascending=False)
    # print("Feature scores based on Mutual info:")
    # print(feature_scores[:10])
