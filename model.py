import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.discriminant_analysis import StandardScaler
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.feature_selection import SelectKBest, VarianceThreshold, f_classif, mutual_info_classif
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix, precision_score, recall_score, accuracy_score
from sklearn.model_selection import GridSearchCV, cross_val_score, train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import make_pipeline
from sklearn.svm import SVC
import xgboost as xgb
from xgboost import XGBClassifier
from xgboost.callback import EarlyStopping



def split_test_data(df, target_column="diagnosis", test_size=0.2):
    features = df.drop(columns=[target_column])
    labels = df[target_column]
    
    # Perform the split
    X_train, X_test, y_train, y_test = train_test_split(
        features, labels, test_size=test_size, random_state=42
    )
    
    # Create DataFrames using original indices
    train_df = df.loc[X_train.index].copy()
    test_df = df.loc[X_test.index].copy()
    
    # Convert to numpy arrays
    X_train = X_train.to_numpy()
    X_test = X_test.to_numpy()
    y_train = y_train.to_numpy()
    y_test = y_test.to_numpy()

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
    
    return features_transformed, labels, features_clean_df, selector.get_support()

def drop_features(df, support_mask):
    support_mask = np.array(list(support_mask)+[True])  # Keep the target column
    df = df.loc[:, support_mask]  
    features = df.drop(columns=["diagnosis"]).values
    labels = df["diagnosis"].values

    return features, labels, df

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

def plot_CV_bar(x, results):

    mean_score = results.mean()
    std_score = results.std()

    # Plot a single bar with appropriate text
    bar = plt.bar([x], [mean_score], yerr=[std_score * 2], capsize=5, 
                  alpha=0.7, color='skyblue', edgecolor='navy')[0]

    # Add value label on the bar
    plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height()/2 + 0.1*np.random.rand(),
             f'{mean_score:.2f}', ha='center', va='bottom', fontsize=9)

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

def plot_GridCV_heatmap(results_df, name):
    param_columns = [col for col in results_df.columns if col.startswith('param_')]
    if len(param_columns) < 2:
        raise ValueError("GridSearchCV results must have at least two hyperparameters to plot a heatmap.")

    param1, param2 = param_columns[:2]  # Use the first two hyperparameters

    # Pivot the DataFrame to create a matrix for the heatmap
    heatmap_data = results_df.pivot_table(
        index=param1, 
        columns=param2, 
        values='mean_test_score'
    )

    # Plot the heatmap
    plt.figure(figsize=(10, 8))
    ax =sns.heatmap(heatmap_data, annot=True, fmt='.2f', cmap='viridis', cbar=True)
    ax.set_xticklabels([f"{float(label):.3f}" for label in heatmap_data.columns], rotation=45, ha='right')
    ax.set_yticklabels([f"{float(label):.3f}" for label in heatmap_data.index], rotation=0)
    plt.title(f'Grid Search Heatmap - SVC ({name})')
    plt.xlabel(param2)
    plt.ylabel(param1)
    plt.tight_layout()
    plt.show()
    
if __name__ == "__main__":

    verbose = False

    np.random.seed(24)
    # Load your data
    df = pd.read_csv("./data_arrhythmia.csv", na_values=['?'])

    df = handle_missing_data(df, verbose=verbose)
    _, _, df, _ = remove_constant_features(df, "diagnosis", threshold=0)

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

    X_train, X_test, y_train, y_test, train_df, test_df = split_test_data(df)

    X_train, y_train, train_df, features_support = remove_constant_features(train_df, "diagnosis", threshold=0)
    X_test, y_test, test_df = drop_features(test_df, features_support)

    y_train_bin = np.where(y_train > 1, 1, 0)
    y_test_bin = np.where(y_test > 1, 1, 0)

    #Logistic Regression----------------------------------------------------------
    
    # verbose = False

    # Ks = range(99, 160, 10)#range(1,250, 49)

    # if verbose:
    #     print("<====== LOGISTIC REG. ======>")
    #     plt.figure(figsize=(10, 6))

    # best_model = (0,None)
    # best_results = 0.

    # for k in Ks:
    #     df2, features_selected = select_top_features(X_train, y_train_bin, train_df, heuristic="ANOVA", k=k, verbose=False)

    #     param_grid = {
    #         'logisticregression__C': np.linspace(0.01, 0.4, 100)#[0.001, 0.01, 0.05, 0.1, 0.2, 0.5, 0.8, 1, 5, 10, ]  
    #     }
    #     clf = LogisticRegression(random_state=42, penalty='l2', max_iter=1000)
    #     pipe = make_pipeline(StandardScaler(), clf)

    #     grid_search = GridSearchCV(estimator=pipe, param_grid=param_grid, cv=5)
    #     grid_search.fit(features_selected, y_train_bin)
    #     results_df = pd.DataFrame(grid_search.cv_results_)

    #     if best_results <= grid_search.best_score_:
    #         best_results = grid_search.best_score_
    #         best_model = (k, grid_search.best_estimator_)

    #     if verbose:
    #         plot_GridCV_line(results_df, name=f"k = {k}", best = False)

    # if verbose:
    #     plt.xlabel('$C$', fontsize=12)
    #     plt.ylabel('Mean CV Score', fontsize=12)
    #     plt.title('Grid Search Results - Logistic Regression', fontsize=14)
    #     plt.grid(True, alpha=0.3)
    #     plt.legend()
    #     plt.tight_layout()
    #     plt.show()

    # #test best model
    # k_opt, clf = best_model
    # df2, features_selected = select_top_features(X_train, y_train_bin, train_df, heuristic="ANOVA", k=k_opt, verbose=False)
    # mask = np.array([True if col in df2.columns else False for col in test_df.columns])
    # features_test = X_test[:, mask[:-1]]
    # if verbose:
    #     print(f"Best k: {k_opt}, Best CV Score: {best_results:.4f}")
    #     print(f"Best C: {clf.named_steps['logisticregression'].C:.4f}")
    #     clf.fit(features_selected, y_train_bin)
    #     plot_cf_matrix(features_selected, y_train_bin, clf, class_names=["Normal", "Arrhythmia"])
    # test_model_performance(features_selected, features_test, y_train_bin, y_test_bin, clf, verbose=verbose)
    

    #Naive Bayes------------------------------------------------------------------

    # verbose = False

    # Ks = range(1,250,1)#[1,2,3,5, 10, 20, 50, 100, 200 ]

    # if verbose:
    #     print("<======= NAIVE BAYES =======>")
    #     plt.figure(figsize=(10, 6))

    # best_model = 0
    # best_results = 0.

    # for k in Ks:
    #     df2, features_selected = select_top_features(X_train, y_train_bin, train_df, heuristic="ANOVA", k=k, verbose=False)

    #     clf = GaussianNB()
    #     scores = cross_val_score(clf, features_selected, y_train_bin, cv=5)

    #     if best_results <= scores.mean():
    #         best_results = scores.mean()
    #         best_model = k

    #     if verbose:
    #         plot_CV_bar(k, scores)

    # if verbose:
    #     plt.grid(True, alpha=0.3, axis='y')
    #     plt.xlabel('$k$', fontsize=12)
    #     plt.ylabel('Mean CV Score', fontsize=12)
    #     plt.title('Grid Search Results - Naive Bayes', fontsize=14)
    #     plt.legend()
    #     plt.tight_layout()
    #     plt.show()

    # #test best model
    # k_opt, clf = best_model, GaussianNB()
    # df2, features_selected = select_top_features(X_train, y_train_bin, train_df, heuristic="ANOVA", k=k_opt, verbose=False)
    # mask = np.array([True if col in df2.columns else False for col in test_df.columns])
    # features_test = X_test[:, mask[:-1]]
    # if verbose:
    #     print(f"Best k: {k_opt}, Best CV Score: {best_results:.4f}")
    #     clf.fit(features_selected, y_train_bin)
    #     plot_cf_matrix(features_selected, y_train_bin, clf, class_names=["Normal", "Arrhythmia"])
    # test_model_performance(features_selected, features_test, y_train_bin, y_test_bin, clf, verbose=verbose)


    #Support Vector Machine-------------------------------------------------------

    # verbose = False

    # kernels = ['rbf', 'poly', 'sigmoid']
    # K = 130
    
    # if verbose:
    #     print("<=========== SVM ===========>")

    # best_model = (0,None)
    # best_results = 0.

    # df2, features_selected = select_top_features(X_train, y_train_bin, train_df, heuristic="ANOVA", k=K, verbose=False)

    # param_grid = {
    #     'rbf' :     { 'svc__C': np.linspace(0.01, 10, 30), 'svc__gamma': np.linspace(0.0001, 0.03, 30)  },
    #     'poly' :    { 'svc__C': np.linspace(0.01, 10, 30), 'svc__degree': [2, 3, 4, 5]              },
    #     'sigmoid' : { 'svc__C': np.linspace(0.01, 10, 30), 'svc__gamma': np.linspace(0.0001, 0.03, 30)  }
    # }

    # for kernel in kernels:

    #     clf = SVC(kernel=kernel, max_iter=10000)
    #     pipe = make_pipeline(StandardScaler(), clf)

    #     grid_search = GridSearchCV(estimator=pipe, param_grid=param_grid[kernel], cv=5)
    #     grid_search.fit(features_selected, y_train_bin)
    #     results_df = pd.DataFrame(grid_search.cv_results_)

    #     if best_results <= grid_search.best_score_:
    #         best_results = grid_search.best_score_
    #         best_model = (kernel, grid_search.best_estimator_)

    #     if verbose:
    #         plot_GridCV_heatmap(results_df, name=kernel)

    # #test best model
    # kernel_opt, clf = best_model

    # mask = np.array([True if col in df2.columns else False for col in test_df.columns])
    # features_test = X_test[:, mask[:-1]]
    # if verbose:
    #     print(f"Best kernel: {kernel_opt}, Best CV Score: {best_results:.4f}")
    #     print(f"Best C: {clf.named_steps['svc'].C:.4f}")
    #     if kernel_opt in ['rbf', 'sigmoid']:
    #         print(f"Best gamma: {clf.named_steps['svc'].gamma:.4f}")
    #     else:
    #         print(f"Best degree: {clf.named_steps['svc'].degree:.4f}")

    #     clf.fit(features_selected, y_train_bin)
    #     plot_cf_matrix(features_selected, y_train_bin, clf, class_names=["Normal", "Arrhythmia"])
    # test_model_performance(features_selected, features_test, y_train_bin, y_test_bin, clf, verbose=verbose)


    #Ensamble Methods-------------------------------------------------------------

    verbose = True

    #K = 120
    Ks = [20, 25, 30]#,35,40,45,50,55,60]#[50, 75, 100, 150, 200, 250]
    if verbose:
        print("<=========== XGBoost ===========>")
    
    best_model = None
    best_result = 0.

    for K in Ks:
    
        X_train_split, X_val_split, y_train_split, y_val_split, train_df_split, val_df_split \
            = split_test_data(train_df, target_column="diagnosis", test_size=0.2)
        

        X_train_split, y_train_split, train_df_split, features_support = remove_constant_features(train_df_split, "diagnosis", threshold=0)
        X_val_split, y_val_split, val_df_split = drop_features(val_df_split, features_support)
        X_test_split, y_test_split, test_df_split = drop_features(test_df, features_support)
        
        y_train_split_bin = np.where(y_train_split > 1, 1, 0)
        y_val_split_bin = np.where(y_val_split > 1, 1, 0)

        df2, features_selected = select_top_features(X_train_split, y_train_split, train_df_split, heuristic="ANOVA", k=K, verbose=False)
        mask = np.array([True if col in df2.columns else False for col in test_df_split.columns])

        features_test = X_test_split[:, mask[:-1]]
        features_val = X_val_split[:, mask[:-1]]


        # Model

        # param_grid = {
        #     "max_depth": [3, 4, 5, 6, 7],
        #     "learning_rate": [0.02, 0.05, 0.1],
        #     "subsample": [0.7, 0.9, 0.99],
        #     "colsample_bytree": [0.7,0.85, 1.0],
        #     "reg_alpha": [0.2, 0.5],
        #     "reg_lambda": [0.05, 0.2, 1 ]
        # }
        param_grid = {
            "max_depth": [ 4,5, 6, 7],
            "learning_rate": [ 0.01, 0.04, 0.09, 0.2, 0.3],
            "subsample": [0.9, 0.95],
            "colsample_bytree": [0.85,0.9, 1.0],
            "reg_alpha": [0.2,0.35, 0.5,0.6, 0.7],
            "reg_lambda": [0.04,0.05,0.07,0.09,0.11]
        }


        # param_grid = {
        #     "max_depth": [3,4,5,6],
        #     "learning_rate": [0.03,0.05,0.1],
        #     "subsample": [0.7,0.9],
        #     "colsample_bytree": [0.7,0.9],
        #     "reg_alpha": [0,0.1,0.5],
        #     "reg_lambda": [1,5,10]
        # }
        # param_grid = {'colsample_bytree': [0.7], 'learning_rate':[0.1], 'max_depth': [5], 'reg_alpha': [0.5], 'reg_lambda': [1], 'subsample': [0.9]}


        model = XGBClassifier(
            n_estimators=2000,#1000
            # max_depth=3,
            # max_leaves=3, 
            # reg_alpha = 0.1,
            # reg_lambda = 0.05,
            # gamma = 0,
            # min_child_weight = 1,
            # learning_rate=0.01,
            eval_metric="error",
            tree_method="hist",
            early_stopping_rounds=70#40
        )

        grid = GridSearchCV(
            model,
            param_grid,
            scoring="accuracy",
            cv=5,
            verbose=1,
            n_jobs=-1
        )

        
        # df2, features_selected = select_top_features(X_train, y_train_bin, train_df, heuristic="ANOVA", k=K, verbose=False) 
        # print(features_selected.shape)
        # mask = np.array([True if col in df2.columns else False for col in test_df.columns])
        # features_test = X_test[:, mask[:-1]]

        grid.fit(features_selected, y_train_split_bin,
                eval_set=[(features_val, y_val_split_bin)],
                verbose=0)
        print(f'K = {K}')
    # scores = cross_val_score(model, features_selected, y_train_split_bin, cv=5, scoring="accuracy")
        print('\t', grid.best_params_)

        print('\t', grid.best_score_)

        if best_result <= grid.best_score_:
            best_result = grid.best_score_
            best_model = (K, grid.best_estimator_.best_iteration, grid.best_estimator_)
    
    
    print(best_model)
    Kopt, iter, best_model = best_model
    # # Or if you want to be explicit:
    # best_model.set_params(n_estimators=15)

    # test_model_performance(features_selected, features_test, y_train_bin, y_test_bin, best_model, verbose=True)
    # test_model_performance(features_selected, features_test, y_train_split_bin, y_test_bin, best_model, verbose=True)
    


    # Train
    best_model.fit(
        features_selected,
        y_train_split_bin,
        eval_set=[
            (features_selected, y_train_split_bin),
            # (features_test, y_test_bin)
            (features_val, y_val_split_bin)
        ],
        verbose=False
    )

    # Results
    results = best_model.evals_result()
    epochs = len(results['validation_0']['error'])
    x_axis = range(epochs)

    # Plot
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(x_axis, results['validation_0']['error'], label='Train')
    ax.plot(x_axis, results['validation_1']['error'], label='Validation')

    ax.legend()
    plt.ylabel('Accuracy Score')
    plt.xlabel('Iterations (n_estimators)')
    plt.title('XGBoost Training Process')




    y_pred = best_model.predict(features_test)
    acc = accuracy_score(y_test_bin, y_pred)
    precision = precision_score(y_test_bin, y_pred, zero_division=0)
    recall = recall_score(y_test_bin, y_pred, zero_division=0)
    f1 = 2 * (precision * recall) / (precision + recall + 1e-10)
    if verbose:
        print(f"//=====TEST SCORES====\\\\")
        print(f"|| Accuracy : {acc:.4f}  ||")
        print(f"|| Precision: {precision:.4f}  ||")
        print(f"|| Recall   : {recall:.4f}  ||")
        print(f"|| F1 Score : {f1:.4f}  ||")
        print(f"\\\\====================//")

    

    # bst = xgb.train(params, dtrain, num_round, evals=evallist, evals_result=evals_result)

    # # Plotting the metrics (example assumes 'validation_0' and 'logloss' in evals_result)
    # plt.plot(evals_result['validation_0']['logloss'], label='train loss')
    # plt.plot(evals_result['validation_1']['logloss'], label='validation loss')
    # plt.legend()
    # plt.show()
    
    plt.show()
    best_model.set_params(n_estimators=iter)
    best_model.set_params(early_stopping_rounds = None)
    df2, features_selected = select_top_features(X_train, y_train_bin, train_df, heuristic="ANOVA", k=Kopt, verbose=False)
    test_model_performance(features_selected, features_test, y_train_bin, y_test_bin, best_model, verbose=True)

    #MULTICLASS PROBLEM


    # best_features = SelectKBest(score_func=mutual_info_classif, k=10)
    # fit = best_features.fit(features, labels_binary)
    # feature_scores = pd.DataFrame({'Feature': df.columns[:-1], 'Score': fit.scores_}).sort_values(by='Score', ascending=False)
    # print("Feature scores based on Mutual info:")
    # print(feature_scores[:10])
