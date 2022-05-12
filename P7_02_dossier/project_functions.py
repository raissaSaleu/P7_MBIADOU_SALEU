############### PROJET 7 #################

import time
import pandas as pd
import numpy as np
import string as st
import re
import os
from scipy import stats
from scipy.spatial import distance
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib.cm as cm
import seaborn as sns
import warnings

import gc
import time
from contextlib import contextmanager
from lightgbm import LGBMClassifier
from sklearn.metrics import roc_auc_score, fbeta_score, roc_curve, confusion_matrix
from sklearn.model_selection import KFold, StratifiedKFold


def describe_dataset(source_files):
    '''
        Outputs a presentation pandas dataframe for the dataset.

        Parameters
        ----------------
        sourceFiles     : dict with :
                            - keys : the names of the files
                            - values : the dataframe for the data
    
        Returns
        ---------------
        presentation_df : pandas dataframe :
                            - a column "Nom du fichier" : the name of the file
                            - a column "Nb de lignes"   : the number of rows per file
                            - a column "Nb de colonnes" : the number of columns per file
    '''

    print("Les données se décomposent en {} fichier(s): \n".format(len(source_files)))

    filenames = []
    files_nb_lines = []
    files_nb_columns = []
    nan_percent = []
    duplicate_percent = []

    for filename, file_data in source_files.items():
        filenames.append(filename)
        files_nb_lines.append(len(file_data))
        files_nb_columns.append(len(file_data.columns))
        nan_percent.append(round(file_data.isna().sum().sum()/file_data.size*100, 2))
        duplicate_percent.append(round(file_data.duplicated().sum().sum()/file_data.size*100, 2))

                           
    # Create a dataframe for presentation purposes
    presentation_df = pd.DataFrame({'Nom du fichier':filenames,
                                    'Nb de lignes':files_nb_lines,
                                    'Nb de colonnes':files_nb_columns,
                                    '%NaN' :nan_percent,
                                    '%Duplicate' :duplicate_percent})

    presentation_df.index += 1

    return presentation_df

#------------------------------------------

def missingdata(data, name, larg, long):
    total = data.isnull().sum().sort_values(ascending = False)
    percent = (data.isnull().sum()/data.isnull().count()*100).sort_values(ascending = False)
    ms=pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
    ms= ms[ms["Percent"] > 0]
    f,ax =plt.subplots(figsize=(larg,long))
    plt.xticks(rotation='90')
    fig=sns.barplot(ms.index, ms["Percent"])
    plt.xlabel('Colonnes', fontsize=15)
    plt.ylabel('% valeurs manquantes', fontsize=15)
    plt.title('Pourcentage de valeurs manquantes ('+name+')', fontsize=22, fontweight='bold')
    #ms= ms[ms["Percent"] > 0]
    #return ms

#------------------------------------------

# function for plotting repetitive countplots in univariate categorical analysis on applicationDF
# This function will create two subplots: 
# 1. Count plot of categorical column w.r.t TARGET; 
# 2. Percentage of defaulters within column

def univariate_categorical(applicationDF,feature,titre,ylog=False,label_rotation=False,
                           horizontal_layout=True):
    temp = applicationDF[feature].value_counts()
    df1 = pd.DataFrame({feature: temp.index,'Number of contracts': temp.values})

    # Calculate the percentage of target=1 per category value
    cat_perc = applicationDF[[feature, 'TARGET']].groupby([feature],as_index=False).mean()
    cat_perc["TARGET"] = cat_perc["TARGET"]*100
    cat_perc.sort_values(by='TARGET', ascending=False, inplace=True)
    
    if(horizontal_layout):
        fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(12,5))
    else:
        fig, (ax1, ax2) = plt.subplots(nrows=2, figsize=(20,24))
        
    # 1. Subplot 1: Count plot of categorical column
    # sns.set_palette("Set2")
    s = sns.countplot(ax=ax1, 
                    x = feature, 
                    data=applicationDF,
                    hue ="TARGET",
                    order=cat_perc[feature],
                    palette=['g','r'])
        
    # Define common styling
    ax1.set_title(titre, fontdict={'fontsize' : 15, 'fontweight' : 'bold'}) 
    ax1.legend(['Remboursé','Défaillant'])
    
    # If the plot is not readable, use the log scale.
    if ylog:
        ax1.set_yscale('log')
        ax1.set_ylabel("Count (log)",fontdict={'fontsize' : 15, 'fontweight' : 'bold'})   
    
    
    if(label_rotation):
        s.set_xticklabels(s.get_xticklabels(),rotation=90)
    
    # 2. Subplot 2: Percentage of defaulters within the categorical column
    s = sns.barplot(ax=ax2, 
                    x = feature, 
                    y='TARGET', 
                    order=cat_perc[feature], 
                    data=cat_perc,
                    palette='Set2')
    
    if(label_rotation):
        s.set_xticklabels(s.get_xticklabels(),rotation=90)
    plt.ylabel('Pourcentage de défaillants [%]', fontsize=10)
    plt.tick_params(axis='both', which='major', labelsize=10)
    ax2.set_title(titre+" (% Défaillants)", fontdict={'fontsize' : 15, 'fontweight' : 'bold'}) 

    plt.show();

#------------------------------------------

def plot_distribution(applicationDF,feature, title):
    plt.figure(figsize = (10, 4))

    t0 = applicationDF.loc[applicationDF['TARGET'] == 0]
    t1 = applicationDF.loc[applicationDF['TARGET'] == 1]

    
    sns.kdeplot(t0[feature].dropna(), label = 'Remboursé', color='g')
    sns.kdeplot(t1[feature].dropna(), label = 'Défaillant', color='r')
    plt.title(title, fontsize='20', fontweight='bold')
    #plt.ylabel('Density',fontsize='14')
    #plt.xlabel(fontsize='14')
    plt.legend()
    plt.show()   
    
#------------------------------------------

#------------------------------------------
# PREPROCESSING AND FEATURES INGENEERING
#------------------------------------------

@contextmanager
def timer(title):
    t0 = time.time()
    yield
    print("{} - done in {:.0f}s".format(title, time.time() - t0))
#------------------------------------------    
    
# One-hot encoding for categorical columns with get_dummies
def one_hot_encoder(df, nan_as_category = True):
    original_columns = list(df.columns)
    categorical_columns = [col for col in df.columns if df[col].dtype == 'object']
    df = pd.get_dummies(df, columns= categorical_columns, dummy_na= nan_as_category)
    new_columns = [c for c in df.columns if c not in original_columns]
    return df, new_columns
#------------------------------------------

# Preprocess application_train.csv and application_test.csv
def application_train_test(PATH, num_rows = None, nan_as_category = False):
    # Read data and merge
    df = pd.read_csv(PATH+'/application_train.csv', nrows= num_rows)
    test_df = pd.read_csv(PATH+'/application_test.csv', nrows= num_rows)
    print("Train samples: {}, test samples: {}".format(len(df), len(test_df)))
    df = df.append(test_df).reset_index()
    # Optional: Remove 4 applications with XNA CODE_GENDER (train set)
    df = df[df['CODE_GENDER'] != 'XNA']
    
    # Categorical features with Binary encode (0 or 1; two categories)
    for bin_feature in ['CODE_GENDER', 'FLAG_OWN_CAR', 'FLAG_OWN_REALTY']:
        df[bin_feature], uniques = pd.factorize(df[bin_feature])
    # Categorical features with One-Hot encode
    df, cat_cols = one_hot_encoder(df, nan_as_category)
    
    # NaN values for DAYS_EMPLOYED: 365.243 -> nan
    df['DAYS_EMPLOYED'].replace(365243, np.nan, inplace= True)
    # Some simple new features (percentages)
    df['DAYS_EMPLOYED_PERC'] = df['DAYS_EMPLOYED'] / df['DAYS_BIRTH']
    df['INCOME_CREDIT_PERC'] = df['AMT_INCOME_TOTAL'] / df['AMT_CREDIT']
    df['INCOME_PER_PERSON'] = df['AMT_INCOME_TOTAL'] / df['CNT_FAM_MEMBERS']
    df['ANNUITY_INCOME_PERC'] = df['AMT_ANNUITY'] / df['AMT_INCOME_TOTAL']
    df['PAYMENT_RATE'] = df['AMT_ANNUITY'] / df['AMT_CREDIT']
    del test_df
    gc.collect()
    return df

#------------------------------------------

# Preprocess bureau.csv and bureau_balance.csv
def bureau_and_balance(PATH, num_rows = None, nan_as_category = True):
    bureau = pd.read_csv(PATH+'/bureau.csv', nrows = num_rows)
    bb = pd.read_csv(PATH+'/bureau_balance.csv', nrows = num_rows)
    bb, bb_cat = one_hot_encoder(bb, nan_as_category)
    bureau, bureau_cat = one_hot_encoder(bureau, nan_as_category)
    
    # Bureau balance: Perform aggregations and merge with bureau.csv
    bb_aggregations = {'MONTHS_BALANCE': ['min', 'max', 'size']}
    for col in bb_cat:
        bb_aggregations[col] = ['mean']
    bb_agg = bb.groupby('SK_ID_BUREAU').agg(bb_aggregations)
    bb_agg.columns = pd.Index([e[0] + "_" + e[1].upper() for e in bb_agg.columns.tolist()])
    bureau = bureau.join(bb_agg, how='left', on='SK_ID_BUREAU')
    bureau.drop(['SK_ID_BUREAU'], axis=1, inplace= True)
    del bb, bb_agg
    gc.collect()
    
    # Bureau and bureau_balance numeric features
    num_aggregations = {
        'DAYS_CREDIT': ['min', 'max', 'mean', 'var'],
        'DAYS_CREDIT_ENDDATE': ['min', 'max', 'mean'],
        'DAYS_CREDIT_UPDATE': ['mean'],
        'CREDIT_DAY_OVERDUE': ['max', 'mean'],
        'AMT_CREDIT_MAX_OVERDUE': ['mean'],
        'AMT_CREDIT_SUM': ['max', 'mean', 'sum'],
        'AMT_CREDIT_SUM_DEBT': ['max', 'mean', 'sum'],
        'AMT_CREDIT_SUM_OVERDUE': ['mean'],
        'AMT_CREDIT_SUM_LIMIT': ['mean', 'sum'],
        'AMT_ANNUITY': ['max', 'mean'],
        'CNT_CREDIT_PROLONG': ['sum'],
        'MONTHS_BALANCE_MIN': ['min'],
        'MONTHS_BALANCE_MAX': ['max'],
        'MONTHS_BALANCE_SIZE': ['mean', 'sum']
    }
    # Bureau and bureau_balance categorical features
    cat_aggregations = {}
    for cat in bureau_cat: cat_aggregations[cat] = ['mean']
    for cat in bb_cat: cat_aggregations[cat + "_MEAN"] = ['mean']
    
    bureau_agg = bureau.groupby('SK_ID_CURR').agg({**num_aggregations, **cat_aggregations})
    bureau_agg.columns = pd.Index(['BURO_' + e[0] + "_" + e[1].upper() for e in bureau_agg.columns.tolist()])
    # Bureau: Active credits - using only numerical aggregations
    active = bureau[bureau['CREDIT_ACTIVE_Active'] == 1]
    active_agg = active.groupby('SK_ID_CURR').agg(num_aggregations)
    active_agg.columns = pd.Index(['ACTIVE_' + e[0] + "_" + e[1].upper() for e in active_agg.columns.tolist()])
    bureau_agg = bureau_agg.join(active_agg, how='left', on='SK_ID_CURR')
    del active, active_agg
    gc.collect()
    # Bureau: Closed credits - using only numerical aggregations
    closed = bureau[bureau['CREDIT_ACTIVE_Closed'] == 1]
    closed_agg = closed.groupby('SK_ID_CURR').agg(num_aggregations)
    closed_agg.columns = pd.Index(['CLOSED_' + e[0] + "_" + e[1].upper() for e in closed_agg.columns.tolist()])
    bureau_agg = bureau_agg.join(closed_agg, how='left', on='SK_ID_CURR')
    del closed, closed_agg, bureau
    gc.collect()
    return bureau_agg

#------------------------------------------

# Preprocess previous_applications.csv
def previous_applications(PATH, num_rows = None, nan_as_category = True):
    prev = pd.read_csv(PATH+'/previous_application.csv', nrows = num_rows)
    prev, cat_cols = one_hot_encoder(prev, nan_as_category= True)
    # Days 365.243 values -> nan
    prev['DAYS_FIRST_DRAWING'].replace(365243, np.nan, inplace= True)
    prev['DAYS_FIRST_DUE'].replace(365243, np.nan, inplace= True)
    prev['DAYS_LAST_DUE_1ST_VERSION'].replace(365243, np.nan, inplace= True)
    prev['DAYS_LAST_DUE'].replace(365243, np.nan, inplace= True)
    prev['DAYS_TERMINATION'].replace(365243, np.nan, inplace= True)
    # Add feature: value ask / value received percentage
    prev['APP_CREDIT_PERC'] = prev['AMT_APPLICATION'] / prev['AMT_CREDIT']
    # Previous applications numeric features
    num_aggregations = {
        'AMT_ANNUITY': ['min', 'max', 'mean'],
        'AMT_APPLICATION': ['min', 'max', 'mean'],
        'AMT_CREDIT': ['min', 'max', 'mean'],
        'APP_CREDIT_PERC': ['min', 'max', 'mean', 'var'],
        'AMT_DOWN_PAYMENT': ['min', 'max', 'mean'],
        'AMT_GOODS_PRICE': ['min', 'max', 'mean'],
        'HOUR_APPR_PROCESS_START': ['min', 'max', 'mean'],
        'RATE_DOWN_PAYMENT': ['min', 'max', 'mean'],
        'DAYS_DECISION': ['min', 'max', 'mean'],
        'CNT_PAYMENT': ['mean', 'sum'],
    }
    # Previous applications categorical features
    cat_aggregations = {}
    for cat in cat_cols:
        cat_aggregations[cat] = ['mean']
    
    prev_agg = prev.groupby('SK_ID_CURR').agg({**num_aggregations, **cat_aggregations})
    prev_agg.columns = pd.Index(['PREV_' + e[0] + "_" + e[1].upper() for e in prev_agg.columns.tolist()])
    # Previous Applications: Approved Applications - only numerical features
    approved = prev[prev['NAME_CONTRACT_STATUS_Approved'] == 1]
    approved_agg = approved.groupby('SK_ID_CURR').agg(num_aggregations)
    approved_agg.columns = pd.Index(['APPROVED_' + e[0] + "_" + e[1].upper() for e in approved_agg.columns.tolist()])
    prev_agg = prev_agg.join(approved_agg, how='left', on='SK_ID_CURR')
    # Previous Applications: Refused Applications - only numerical features
    refused = prev[prev['NAME_CONTRACT_STATUS_Refused'] == 1]
    refused_agg = refused.groupby('SK_ID_CURR').agg(num_aggregations)
    refused_agg.columns = pd.Index(['REFUSED_' + e[0] + "_" + e[1].upper() for e in refused_agg.columns.tolist()])
    prev_agg = prev_agg.join(refused_agg, how='left', on='SK_ID_CURR')
    del refused, refused_agg, approved, approved_agg, prev
    gc.collect()
    return prev_agg

#------------------------------------------

# Preprocess POS_CASH_balance.csv
def pos_cash(PATH, num_rows = None, nan_as_category = True):
    pos = pd.read_csv(PATH+'/POS_CASH_balance.csv', nrows = num_rows)
    pos, cat_cols = one_hot_encoder(pos, nan_as_category= True)
    # Features
    aggregations = {
        'MONTHS_BALANCE': ['max', 'mean', 'size'],
        'SK_DPD': ['max', 'mean'],
        'SK_DPD_DEF': ['max', 'mean']
    }
    for cat in cat_cols:
        aggregations[cat] = ['mean']
    
    pos_agg = pos.groupby('SK_ID_CURR').agg(aggregations)
    pos_agg.columns = pd.Index(['POS_' + e[0] + "_" + e[1].upper() for e in pos_agg.columns.tolist()])
    # Count pos cash accounts
    pos_agg['POS_COUNT'] = pos.groupby('SK_ID_CURR').size()
    del pos
    gc.collect()
    return pos_agg

#------------------------------------------

# Preprocess installments_payments.csv
def installments_payments(PATH, num_rows = None, nan_as_category = True):
    ins = pd.read_csv(PATH+'/installments_payments.csv', nrows = num_rows)
    ins, cat_cols = one_hot_encoder(ins, nan_as_category= True)
    # Percentage and difference paid in each installment (amount paid and installment value)
    ins['PAYMENT_PERC'] = ins['AMT_PAYMENT'] / ins['AMT_INSTALMENT']
    ins['PAYMENT_DIFF'] = ins['AMT_INSTALMENT'] - ins['AMT_PAYMENT']
    # Days past due and days before due (no negative values)
    ins['DPD'] = ins['DAYS_ENTRY_PAYMENT'] - ins['DAYS_INSTALMENT']
    ins['DBD'] = ins['DAYS_INSTALMENT'] - ins['DAYS_ENTRY_PAYMENT']
    ins['DPD'] = ins['DPD'].apply(lambda x: x if x > 0 else 0)
    ins['DBD'] = ins['DBD'].apply(lambda x: x if x > 0 else 0)
    # Features: Perform aggregations
    aggregations = {
        'NUM_INSTALMENT_VERSION': ['nunique'],
        'DPD': ['max', 'mean', 'sum'],
        'DBD': ['max', 'mean', 'sum'],
        'PAYMENT_PERC': ['max', 'mean', 'sum', 'var'],
        'PAYMENT_DIFF': ['max', 'mean', 'sum', 'var'],
        'AMT_INSTALMENT': ['max', 'mean', 'sum'],
        'AMT_PAYMENT': ['min', 'max', 'mean', 'sum'],
        'DAYS_ENTRY_PAYMENT': ['max', 'mean', 'sum']
    }
    for cat in cat_cols:
        aggregations[cat] = ['mean']
    ins_agg = ins.groupby('SK_ID_CURR').agg(aggregations)
    ins_agg.columns = pd.Index(['INSTAL_' + e[0] + "_" + e[1].upper() for e in ins_agg.columns.tolist()])
    # Count installments accounts
    ins_agg['INSTAL_COUNT'] = ins.groupby('SK_ID_CURR').size()
    del ins
    gc.collect()
    return ins_agg

#------------------------------------------

# Preprocess credit_card_balance.csv
def credit_card_balance(PATH, num_rows = None, nan_as_category = True):
    cc = pd.read_csv(PATH+'/credit_card_balance.csv', nrows = num_rows)
    cc, cat_cols = one_hot_encoder(cc, nan_as_category= True)
    # General aggregations
    cc.drop(['SK_ID_PREV'], axis= 1, inplace = True)
    cc_agg = cc.groupby('SK_ID_CURR').agg(['min', 'max', 'mean', 'sum', 'var'])
    cc_agg.columns = pd.Index(['CC_' + e[0] + "_" + e[1].upper() for e in cc_agg.columns.tolist()])
    # Count credit card lines
    cc_agg['CC_COUNT'] = cc.groupby('SK_ID_CURR').size()
    del cc
    gc.collect()
    return cc_agg

#------------------------------------------

#------------------------------------------
# MODELISATION
#------------------------------------------

def evaluate_model(gs, x, y, x_test, y_test, model_name, balancing_method):
    
    #Entrainement
    start = time.time()
    model = gs.fit(x,y)
    end = time.time()-start

    if (model_name != 'Baseline'):
        df_results = pd.DataFrame.from_dict(model.cv_results_)

    #Training Performance
    if (model_name == 'Baseline'):
        #y_pred = model.predict(x)
        y_proba = model.predict_proba(x)

        auc_train = round(roc_auc_score(y, y_proba[:,1]),3) 
        #f2_train = round(fbeta_score(y, y_pred, beta=2), 3)
    else:
        auc_train = round(model.best_score_,3) 
        #f2_train = round(np.mean(df_results[df_results.rank_test_F2 == 1]['mean_test_F2']),3)

    #Testing Performance
    #y_pred = model.predict(x_test)
    y_proba = model.predict_proba(x_test)
    auc_test = round(roc_auc_score(y_test, y_proba[:,1]),3) 
    #f2_test = round(fbeta_score(y_test, y_pred, beta=2), 3)

    row = [model_name, 
            balancing_method,
            auc_train, 
            auc_test,
            #f2_train,
            #f2_test,
            end]

    return row

#------------------------------------------

def evaluate_model2(model, x, y, x_test, y_test, folds=5, loss_func=None):
    '''
        Uses cross-validation to determine the score of a model 
        on train data, then calculates the score on test data.
    
        Parameters
        --------
        - model     : a machine learning model
        - x         : pandas dataframe
                      The training features
        - y         : pandas dataframe
                      The training labels
        - x_test    : pandas dataframe
                      The test features
        - y_test    : pandas dataframe
                      The test labels
        - scoring   : Cost function
                      The cost function to use for scoring
        - folds     : int
                      The number of folds to use for the cross-validation
        - loss_func : Loss function
                      The loss function to use for the algorithms that allow
                      custom loss functions
            
        Returns
        --------
        -, -, -, - : tuple
                     - The training custom scores for each fold (array)
                     - The custom score for the test data (float)
                     - The training ROC AUC scores for each fold (array)
                     - The ROC AUC score for the test data (float)
    '''

    #cv_custom_scores = []
    cv_auc_scores = []
    cv_f2_scores = []

    y_pred_proba = []

    # create folds
    kf = StratifiedKFold(n_splits=folds)
    
    for train_indices, valid_indices in kf.split(x, y):
        # Training data for the fold
        xtrn, ytrn = x.iloc[train_indices], y.iloc[train_indices]
        # Validation data for the fold
        xval, yval = x.iloc[valid_indices], y.iloc[valid_indices]

        # train
        if loss_func!=None:
            model.fit(xtrn, ytrn, eval_metric = loss_func)
        else:
            model.fit(xtrn, ytrn)

        # predict values on validation set
        ypred = model.predict(xval)
        
        # save probabilities for class 1
        yprob = model.predict_proba(xval)
        y_pred_proba+=(list(yprob[:,1]))

        # calculate and save scores
        auc_score = round(roc_auc_score(yval, ypred), 3)
        cv_auc_scores.append(auc_score)

        f2_score = round(fbeta_score(yval, ypred, beta=2), 3)
        cv_f2_scores.append(f2_score)
        
        #custom_score = round(scoring(yval, ypred), 3)
        #cv_custom_scores.append(custom_score)

    if loss_func!=None:
        model.fit(x, y, eval_metric=loss_func)
        y_pred = model.predict(x_test)
    else:
        model.fit(x, y)
        y_pred = model.predict(x_test)

    auc_score_test = round(roc_auc_score(y_test, y_pred), 3)
    
    f2_score_test = round(fbeta_score(y_test, y_pred, beta=2), 3)

    #custom_score_test = round(scoring(y_test, y_pred), 3)

    return np.array(cv_f2_scores), \
           f2_score_test, \
           np.array(cv_auc_scores), \
           auc_score_test

#------------------------------------------

def plotComparaisonResults(metrics_compare, metric):
    
    fig, ax = plt.subplots()
    
    # create data
    x = np.arange(4)
    y1 = metrics_compare [metrics_compare['Balancing_method'] == "Undersampling"] [metric]
    y2 = metrics_compare [metrics_compare['Balancing_method'] == "Oversampling"] [metric]
    y3 = metrics_compare [metrics_compare['Balancing_method'] == "Balanced"] [metric]
    width = 0.2

    # plot data in grouped manner of bar type
    b1 = plt.bar(x-0.2, y1, width)
    b2 = plt.bar(x, y2, width)
    b3 = plt.bar(x+0.2, y3, width)
    plt.xticks(x, ['Baseline','LinearRegression', 'RandomForest', 'LGBM'])
    
    #if (metric =="F2"):
    #    plt.title('F2-score des modèles (train)')
    
    #if(metric =="F2_test"):
    #    plt.title('F2-score des modèles (test)')

    if (metric =="AUC"):
        plt.title('AUC des modèles (train)')
    
    if(metric =="AUC_test"):
        plt.title('AUC des modèles (test)')
        
    #if (metric =="F2" or metric =="F2_test"):    
     #   plt.ylabel("F2-score")
    if (metric =="Time"):
        plt.ylabel("Time (sec)")
        plt.title("Temps d'exécution du fit")    
    else:
        plt.ylabel("AUC score")
    plt.legend(["Undersampling", "Oversampling", "Balanced"], loc='center left', bbox_to_anchor=(1, 0.5))
    plt.show()
#------------------------------------------


def cf_matrix_roc_auc(model, y_true, y_pred, y_pred_proba, roc_auc, title):
    '''This function will make a pretty plot of 
  an sklearn Confusion Matrix using a Seaborn heatmap visualization + ROC Curve.'''
    fig = plt.figure(figsize=(20,15))
  
    plt.subplot(221)
    cf_matrix = confusion_matrix(y_true, y_pred)
    group_names = ['True Neg', 'False Pos', 'False Neg', 'True Pos']
    group_counts = ["{0:0.0f}".format(value) for value in cf_matrix.flatten()]
    group_percentages = ["{0:.2%}".format(value) for value in cf_matrix.flatten()/np.sum(cf_matrix)]
  
    labels = [f"{v1}\n{v2}\n{v3}" for v1, v2, v3 in zip(group_names,group_counts,group_percentages)]
    labels = np.asarray(labels).reshape(2,2)
    sns.heatmap(cf_matrix, annot=labels, fmt="", cmap='Blues')

    plt.subplot(222)
    fpr,tpr,_ = roc_curve(y_true, y_pred_proba)
    plt.plot(fpr, tpr, color='orange', linewidth=5, label='AUC = %0.4f' %roc_auc)
    plt.plot([0, 1], [0, 1], color='darkblue', linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend()

    fig.suptitle(title, fontsize="30", fontweight="bold")
    plt.show()
    
