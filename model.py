import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.discriminant_analysis import StandardScaler
from sklearn.feature_selection import SelectKBest, VarianceThreshold, f_classif, mutual_info_classif
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix, precision_score, recall_score, accuracy_score
from sklearn.model_selection import GridSearchCV, cross_val_score, train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import make_pipeline
from sklearn.svm import SVC

def split_test_data(df, target_column="diagnosis", test_size=0.2):
    features = df.drop(columns=[target_column])
    labels = df[target_column]
    
    # Perform the split and get indices
    X_train, X_test, y_train, y_test = train_test_split(
        features, labels, test_size=test_size, random_state=42
    )
    
    # Use the indices from the split to create train and test DataFrames
    train_df = df.loc[X_train.index].copy()
    train_df[target_column] = y_train.values

    test_df = df.loc[X_test.index].copy()
    test_df[target_column] = y_test.values

    return X_train, X_test, y_train, y_test, train_df, test_df

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

def test_model_performance(X_train, X_test, y_train, y_test, clf, verbose=False):

    model = clf.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = 2 * (precision * recall) / (precision + recall + 1e-10)

    if verbose:
        print(f"//=====TEST SCORES====\\\\")
        print(f"|| Accuracy : {acc:.4f}  ||")
        print(f"|| Precision: {precision:.4f}  ||")
        print(f"|| Recall   : {recall:.4f}  ||")
        print(f"|| F1 Score : {f1:.4f}  ||")
        print(f"\\\\====================//")

    return acc, precision, recall, f1

def plot_GridCV_bar(results_df):
    plt.figure(figsize=(10, 6))

    x_pos = np.arange(len(results_df))
    C_values = results_df['param_logisticregression__C'].astype(str)
    mean_scores = results_df['mean_test_score']
    std_scores = results_df['std_test_score']

    # Create bars
    bars = plt.bar(x_pos, mean_scores, yerr=std_scores*2, capsize=5, 
                alpha=0.7, color='skyblue', edgecolor='navy')

    # Color the best bar differently
    best_idx = results_df['rank_test_score'].idxmin()
    bars[best_idx].set_color('gold')
    bars[best_idx].set_edgecolor('darkorange')

    plt.xlabel('C (Regularization parameter)', fontsize=12)
    plt.ylabel('Mean CV Score', fontsize=12)
    plt.title('Grid Search Results - Logistic Regression', fontsize=14)
    plt.xticks(x_pos, C_values)
    plt.grid(True, alpha=0.3, axis='y')

    # Add value labels on bars
    for i, (score, bar) in enumerate(zip(mean_scores, bars)):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{score:.3f}', ha='center', va='bottom', fontsize=9)

    plt.tight_layout()
    plt.show()

def plot_GridCV_line(results_df, name, best=False):

    # Extract data
    C_values = results_df['param_logisticregression__C'].astype(float)
    mean_scores = results_df['mean_test_score']
    std_scores = results_df['std_test_score']

    # Plot with error bars
    plt.semilogx(C_values, mean_scores, 'o-', linewidth=2, markersize=8, label=name)
    plt.fill_between(C_values, 
                    mean_scores - std_scores, 
                    mean_scores + std_scores, 
                    alpha=0.2)

    # Highlight best score
    if best:
        best_idx = results_df['rank_test_score'].idxmin()
        plt.plot(C_values.iloc[best_idx], mean_scores.iloc[best_idx], 
                '*', markersize=15, label=f'Best C={C_values.iloc[best_idx]}')



if __name__ == "__main__":

    verbose = False

    np.random.seed(24)
    # Load your data
    df = pd.read_csv("./data_arrhythmia.csv", na_values=['?'])

    df = handle_missing_data(df, verbose=verbose)
    df = remove_constant_features(df, "diagnosis", threshold=0)

    # Plot the distribution of diagnosis labels
    if verbose:
        counts = df["diagnosis"].value_counts()
        counts.plot(kind='bar', title='Distribution of Diagnosis Labels')
        plt.xlabel('Diagnosis')
        plt.ylabel('Number of Instances')
        plt.show()

    


    


#=============================================================================================
#BINARY PROBLEM

    # Distribution of binary classes (Abnormal - 1, Normal - 0)
    bin_labels = np.where(df["diagnosis"].values > 1, 1, 0)
    if verbose:
        print("Distribution of binary labels:")
        print(pd.Series(bin_labels).value_counts())

    X_train, X_test, y_train, y_test, df, test_df = split_test_data(df)
    y_train_bin = np.where(y_train > 1, 1, 0)
    y_test_bin = np.where(y_test > 1, 1, 0)

    #Logistic Regression----------------------------------------------------------
    
    verbose = False

    Ks = range(99, 160, 10)#range(1,250, 49)

    if verbose:
        plt.figure(figsize=(10, 6))

    best_model = (0,None)
    best_results = 0.

    for k in Ks:
        df2, features_selected = select_top_features(X_train, y_train_bin, df, heuristic="ANOVA", k=k, verbose=False)

        param_grid = {
            'logisticregression__C': np.linspace(0.01, 0.4, 100)#[0.001, 0.01, 0.05, 0.1, 0.2, 0.5, 0.8, 1, 5, 10, ]  
        }
        clf = LogisticRegression(random_state=42, penalty='l2', max_iter=1000)
        pipe = make_pipeline(StandardScaler(), clf)

        grid_search = GridSearchCV(estimator=pipe, param_grid=param_grid, cv=5)
        grid_search.fit(features_selected, y_train_bin)
        results_df = pd.DataFrame(grid_search.cv_results_)

        if best_results <= grid_search.best_score_:
            best_results = grid_search.best_score_
            best_model = (k, grid_search.best_estimator_)

        if verbose:
            plot_GridCV_line(results_df, name=f"k = {k}", best = False)

    if verbose:
        plt.xlabel('$C$', fontsize=12)
        plt.ylabel('Mean CV Score', fontsize=12)
        plt.title('Grid Search Results - Logistic Regression', fontsize=14)
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.show()

    #test best model
    k_opt, clf = best_model
    df2, features_selected = select_top_features(X_train, y_train_bin, df, heuristic="ANOVA", k=k_opt, verbose=False)
    best_model = clf.fit(features_selected, y_train_bin)
    if verbose:
        plot_cf_matrix(features_selected, y_train_bin, best_model, class_names=["Normal", "Arrhythmia"])
    test_model_performance(X_train, X_test, y_train_bin, y_test_bin, best_model, verbose=True)

    # results = []
    # Cs = np.linspace(0.01, 0.035, 50)#[0.1, 0.5, 1.0, 2, 5, 10.0]
    # Copt = 0.015

    # for C in Cs:
    #     df2, features_selected = select_top_features(features, labels_binary, df, heuristic="ANOVA", k=k, verbose=False)
    #     pipe = make_pipeline(StandardScaler(), clf)
    #     results.append(cross_val_score(pipe, features_selected, labels_binary, cv=5))

    # results = [res.mean() for res in results]
    # plt.figure()
    # plt.plot(Ks, results, label="C = 0.015")
    # plt.xlabel("Number of selected features (k)")
    # plt.ylabel("Cross-validation score")
    # plt.grid()
    # plt.legend()
    # plt.show()
    # print("Cross-validation scores:")
    # print(results[0].mean())

    # model = pipe.fit(features_selected, labels_binary)
    

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
