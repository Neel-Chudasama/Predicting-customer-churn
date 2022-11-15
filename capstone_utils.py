# import core ds libs
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy import stats
from matplotlib import style
import joblib
import plotly.express as px
import sklearn

#Import Models
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report

#Import modules to judge models
from tempfile import mkdtemp
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, plot_confusion_matrix
from sklearn.metrics import confusion_matrix, recall_score, precision_score, f1_score, roc_curve, roc_auc_score

#Import scaling modules
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import RobustScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from scipy.cluster.hierarchy import linkage, dendrogram
from sklearn.cluster import AgglomerativeClustering
from imblearn.over_sampling import SMOTE



'''
Use this line to reload the library after making changes
import ds_utils
from importlib import reload
reload(ds_utils)
'''


def eda(df):
    """
    getting some basic information about each dataframe
    shape of dataframe i.e. number of rows and columns
    total number of rows with null values
    total number of duplicates
    data types of columns

    Args:
    df (dataframe): dataframe containing the data for analysis
    """
    print()
    print(f"Rows: {df.shape[0]} \t Columns: {df.shape[1]}")
    print()
    
    print(f"Total null rows: {df.isnull().sum().sum()}")
    print(f"Percentage null rows: {round(df.isnull().sum().sum() / df.shape[0] * 100, 2)}%")
    print()
    
    print(f"Total duplicate rows: {df[df.duplicated(keep=False)].shape[0]}")
    print(f"Percentage dupe rows: {round(df[df.duplicated(keep=False)].shape[0] / df.shape[0] * 100, 2)}%")
    print()
    
    print(df.dtypes)
    print("-----\n")
    
    print()
    print("The head of the dataframe is: ")
    display(df.head(5))
    
    print()
    print("The tail of the dataframe is:")
    display(df.tail(5))
    
    print()
    print("Description of the numerical columns is as follows")
    display(df.describe())

def binarise_data(cat_list,df):
    
    """
    This function binarises the the column of data, if the column has two unique values: yes or no then it converts it to 1 and 0 respectively 
    If the column has three unique value then it will binarise the data dependent on the cell values

    Args:
    cat_list: list of columns containing category values that need to be binarised 
    df: dataframe which contains all of the category columns to ensure that they can return values 
    """
    
    
    for cols in cat_list:
        if df[cols].value_counts().shape[0] == 2:
            df[cols] = df[cols].map({"Yes":1,"No":0})
        if cols == 'MultipleLines':
            df[cols] = df[cols].map({"Yes":1,"No":0, "No phone service":0})
        elif cols == 'InternetService' or cols == "Contract":
            pass
        elif df[cols].value_counts().shape[0] == 3:
            df[cols] = df[cols].map({"Yes":1,"No":0, "No internet service":0})
    display(df)

def dummy_variables(lst,df):
    """
    This functions turns the column which holds multiple values into dummy variables and then adds the new columns to the original dataframe

    Args:
    lst: List containing all the multiple value columns that need to be converted into dummy variables
    df (dataframe): dataframe containing the multiple value columns and the appended dataframe will return 
    """
    
    for col in lst:
        df = pd.get_dummies(df, columns=[col])
    display(df)
    return df

def plot_distribution(df, col_name,figsize = (15,10),**kwargs):
    """function that goes through the column, splits the column data between 0 and 1 and shows the distribution between
    churning and not churning
    
    Args:
    df: datframe which holds the data of the column and whether each customer churned or not
    col_name: Column in which the churn behaviour is being plotted and displayed, this should be entered as a string
    figsize: The figure size of the plot is already pre defined unless the user wants a larger plot
    **kwargs: This is used to ensure that this function can be used within a for loop and produce multiple subplots
    """
    
    partial_df = df[[col_name,"Churn"]]
    plots = partial_df.groupby(col_name).value_counts().unstack().plot(kind = "bar", figsize = figsize, color = ['green','red'], **kwargs)
    plt.title("Bar plot demonstrating Churn behaviour in " + col_name + " column")
    plt.ylabel("Count")

    for bar in plots.patches:
        plots.annotate(format(bar.get_height(), '.2f'),
                    (bar.get_x() + bar.get_width() / 2,
                        bar.get_height()), ha='center', va='center',
                    size=9, xytext=(0, 8),
                    textcoords='offset points')  
        
        
def plot_stacked_dist(df, col_name,figsize = (15,10),**kwargs):
    """
    function that goes through the column and produces a stacked bar chart of the churn behaviour for each specific chart or not
    
    Args:
    df: Dataframe that hold the data on churn behaviour and the data on each specific column
    col_name: The specific column that will be studied to display the churn behaviour or not
    figsize: The figure size of the plot is already pre defined unless the user wants a larger plot
    **kwargs: This is used to ensure that this function can be used within a for loop and produce multiple subplots
    """

    partial_df = df[[col_name,"Churn"]]
    partial_df = partial_df.groupby(col_name).value_counts().unstack()
    partial_df = partial_df.reset_index()
    partial_df.plot.bar(x=col_name, stacked=True, figsize = figsize, color = ['cyan','red'], **kwargs) 
    plt.title("Bar plot demonstrating Churn behaviour in " + col_name + " column")
    plt.xticks(rotation = 45)
    plt.ylabel("Count")


def churn_percentage(df,col_name):
    '''
    Function that calculates the churn percentage for each category in the column
    
    Args:
    df: Dataframe that hold the data on churn behaviour and the data on each specific column
    col_name: The specific column that will be studied to display the churn behaviour or not, this should be entered as a string
    '''
    
    if len(df[col_name].value_counts()) == 2:
        partial_df = df[[col_name,"Churn"]].value_counts().unstack()
        percentage = {}
        percentage[partial_df.index[0]] = round(partial_df.iloc[0][1]/sum(partial_df.iloc[0])*100,1)
        percentage[partial_df.index[1]] = round(partial_df.iloc[1][1]/sum(partial_df.iloc[1])*100,1)
        print(col_name)
        print(percentage)
    else:
        partial_df = df[[col_name,"Churn"]].value_counts().unstack()
        percentage = {}
        percentage[partial_df.index[0]] = round(partial_df.iloc[0][1]/sum(partial_df.iloc[0])*100,1)
        percentage[partial_df.index[1]] = round(partial_df.iloc[1][1]/sum(partial_df.iloc[1])*100,1)
        percentage[partial_df.index[2]] = round(partial_df.iloc[2][1]/sum(partial_df.iloc[2])*100,1)
        print(col_name)
        print(percentage)


def heat_map(df):
    '''
    Function that displays the correlation between the column in a dataframe and displays the results in the form of a heatmap 
    
    Args:
    df: Dataframe that hold the data for each column in the dataframe, the dataframe which I want to display the correlation for
    '''

    corr = df.corr()
    plt.figure(figsize=(23, 15), dpi = 300)
    matrix = np.triu(corr)
    sns.heatmap(corr, cmap="coolwarm", vmax=1.0, vmin=-1.0, annot = True, mask = matrix)
    plt.show()

def numerical_subplots(df,lst,rows,columns):

    '''
    Function that displays the distribution of data in numerical columns only. Displays the data in the form of a histogram on a subplot dependent on how many columns and rows of the subplot the user desires.  
    
    Args:
    df: Dataframe that hold the data on each specific column
    lst: A list of the numerical columns in the dataframe which the user would like to display
    rows: This is a numerical input that determines how many rows the subplot should display 
    columns: This is a numerical input that determines how many columns of the subplot should display the data 
    '''

    plt.subplots(rows, columns, figsize=(18, 14))

    count = 1

    for col in lst:
        plt.subplot(rows, columns, count)
        plt.hist(df[col])
        plt.title(col)
        plt.xlabel(col)
        plt.ylabel('Frequency')
        count += 1

    plt.tight_layout()
    plt.show()

def categorical_subplots(df,column_list):

    '''
    Function that displays the distribution of data of categorical columns only. Displays the data in the form of a barplot on a subplot, with the number of 
    rows and columns of the subplot pre-defined. 
    
    Args:
    df: Dataframe that hold the data on each specific column
    column_list: A list of all the columns containing categorical data in the dataframe
    '''

    plt.subplots(5, 3, figsize=(17, 20))

    count = 1

    for col in column_list:
        plt.subplot(5, 3, count)
        sns.barplot(x=df[col].value_counts().index, y = df[col].value_counts())
        plt.title(col)
        plt.xlabel(col)
        plt.xticks(rotation = 45)
        plt.ylabel("Count")
        count += 1

        
    plt.tight_layout()
    plt.show()

def contract_month_averages(df):

    '''
    Function that displays the average number of users for each specific contract - month to month, one year and two year
    
    Args:
    df: Dataframe that hold the data on the contract column

    '''

    month_ave = df[df['Contract']=='Month-to-month']['MonthlyCharges'].mean()
    one_ave = df[df['Contract']=='One year']['MonthlyCharges'].mean()
    two_ave = df[df['Contract']=='Two year']['MonthlyCharges'].mean()

    average_monthly_charge = {'Month to month':month_ave,'One Year':one_ave, 'Two year':two_ave}
    monthly_charge_contract = pd.DataFrame.from_dict([average_monthly_charge])
    display(monthly_charge_contract)

def plot_contract_tenure(df):
    '''
    Function that plots the average tenure of customers for each specific contract type in the form of a horizontal barplot  
    
    Args:
    df: Dataframe that hold the data on each specific column
    column_list: A list of all the columns containing categorical data in the dataframe
    '''

    duration_month = df[df['Contract']=='Month-to-month']['tenure'].mean()
    duration_one = df[df['Contract']=='One year']['tenure'].mean()
    duration_two = df[df['Contract']=='Two year']['tenure'].mean()

    average_duration = {'Month_to_month':duration_month,'One_year':duration_one, 'Two_year':duration_two}
    duration_contract = pd.DataFrame.from_dict([average_duration])
    duration_contract.rename(index={0:'Average Tenure'},inplace=True)
    duration_contract = duration_contract.T

    duration_contract.plot(kind= 'barh', figsize = (8,2), color = 'm')
    plt.xlabel("Average tenure (months)")
    plt.show()

def df_histogram(df, col_name, bins = 50, colormap = 'Accent',figsize = (10,7), **kwargs):

    '''
    Function that displays the column data in the desired dataframe in the form of a histogram, the frequence is dependent on the number of customers(rows) 
    
    Args:
    df: Dataframe that hold the data on each specific column
    col_name: The column in the dataframe which the user wants to plot in the form of a histogram
    colormap: This is the color of the histogram the user desires, it has already been pre-defined as 'Accent'. This must come in the form of a string 
    figsize: The figure size of the plot is already pre defined unless the user wants a larger plot
    **kwargs: This is used to ensure that this function can be used within a for loop and produce multiple subplots
    '''    


    df[col_name].plot(kind = "hist", bins = bins,figsize = figsize , colormap = colormap, **kwargs)
    plt.title("Histogram plot of " + str(col_name))
    plt.axvline(df[col_name].mean(), color='k', linestyle='dashed', linewidth=1)
    plt.xlabel(str(col_name))
    plt.ylabel("Frequency")
    

def boxplot(df, first_var,second_var,):
    '''
    Function that plots the data in the form of a boxplot
    
    Args:
    df: Dataframe that hold the data on each specific column
    first_var: This comes in the form of a string and it is dependent on the user, the data will be displayed on the x-axis and will be a category data
    second_var: This comes in the form of a string and it is dependent on the user, the data will be displayed on the y-axis and will be numerical data
    '''

    plt.figure(figsize = (9,15))
    fig = px.box(df, x=first_var, y=second_var)
    fig.show()

def services_count(df):
    
    '''
    This function produces a bar plot of the customers who paid for specific services provided by Telco
    df: The dataframe with the data filtered to visualise the specific relationship
    '''
    
    yes_vals = []
    column_names =[]
    for cols in df.iloc[:, 9:15].columns:
        column_names.append(cols)
        yes_vals.append(df[cols].value_counts()['Yes'])

    plt.figure(figsize=(10,9))
    sns.barplot(x=column_names,y=yes_vals)
    plt.title('Count of customers who paid for Telco services')
    plt.xlabel("Telco services")
    plt.ylabel("Count")
    plt.show()


def bar_plot_count(df,col_name, colormap = 'Accent'):

    '''
    Function that plots the count of rows for a desired column as a bar plot
    
    Args:
    df: Dataframe that hold the data on each specific column
    col_name: The column in the dataframe which the user wants to plot in the form of a bar plot, this comes in the form of a string 
    colormap: This is the color of the barplot, it has been already been predefined as "Accent" but it can be another colour, must come in the form of a string 
    '''

    plt.figure(figsize = (10,7))
    plots = df[col_name].value_counts().plot(kind = "bar", colormap = colormap)
    plt.title("Bar plot demonstrating distribution of "+ str(col_name)+" column")
    plt.xlabel(str(col_name))
    plt.xticks(rotation = 45)
    plt.ylabel("Count")
    
    for bar in plots.patches:
        plots.annotate(format(bar.get_height(), '.2f'),
                    (bar.get_x() + bar.get_width() / 2,
                        bar.get_height()), ha='center', va='center',
                    size=9, xytext=(0, 8),
                    textcoords='offset points')
    
    plt.show()

def horizontal_bar_plot_count(df,col_name, color = [None],colormap = None ):

    '''
    Function that plots the count of rows for a desired column as a horizontal bar plot
    
    Args:
    df: Dataframe that hold the data on each specific column
    col_name: The column in the dataframe which the user wants to plot in the form of a bar plot, this comes in the form of a string 
    colormap: This is the color of the barplot and must be defined with the use of a string 
    color: This is the color of the barplot, should the user want to choose a specific color for specific bars
    '''
    
    plt.figure(figsize = (10,7))
    plots = df[col_name].value_counts().plot(kind = "barh", color = color, colormap = colormap)
    plt.title("Bar plot demonstrating distribution of "+ str(col_name)+" column")
    plt.ylabel(str(col_name))
    plt.xlabel("Count")   
    plt.show()


def standard_scale_data(X_train,X_test):

    '''
    Function that scales the data inputted into the function with the standard scaler and returns the scaled data
    
    Args:
    X_train: Train data that will be scaled using the standard scaler
    X_test: Test dat that will be scaled using the standard scaler
    '''

    ss = StandardScaler()
    ss.fit(X_train)
    X_train_ss = ss.transform(X_train)
    X_test_ss = ss.transform(X_test)

    return X_train_ss, X_test_ss

def minmax_scale_data(X_train,X_test):
    '''
    Function that scales the data inputted into the function with the minmax scaler and returns the scaled data
    
    Args:
    X_train: Train data that will be scaled using the minmax scaler
    X_test: Test dat that will be scaled using the minmas scaler
    '''
    
    mm = MinMaxScaler()
    mm.fit(X_train)
    X_train_mm = mm.transform(X_train)
    X_test_mm = mm.transform(X_test)

    return X_train_mm, X_test_mm

def minmax_clus_data(X):
    '''
    Function that scales the cluster data inputted into the function with the minmax scaler and returns the scaled data
    
    Args:
    X: Data that will be scaled using the minmax scaler
    '''

    mm = MinMaxScaler()
    mm.fit(X)
    X_mm = mm.transform(X)
    return X_mm

'''Statistical tests'''
def chi_squared_t(lst,df):

    '''
    Function that conducts a chi squared statistical test and displays the output of the test in the form of a dataframe
    
    Args:
    lst: This is a list of the categorical columns which the chi squared statistical test will be conducted on 
    df: The dataframe where the data is held for the chi squared tests to be conducted 
    '''

    chi_sq = {}
    for col in lst:
        df2 = df.groupby(col)['Churn'].value_counts().unstack()
        chi_sq[col] = [stats.chi2_contingency(df2)[0],stats.chi2_contingency(df2)[1]]
    chi_sq_df = pd.DataFrame.from_dict(chi_sq)
    chi_sq_df = chi_sq_df.rename(index={0 : 'Chi Squared value',1:'P-Value'})
    display(chi_sq_df.T)


'''Clustering models'''
def plot_pca(scaled_df, comp):

    '''
    This is a function that the plots the data which has pca imposed on it, the data is displayed as a pairplot
    
    Args:
    scaled_df: The dataframe where the data is held, it has been scaled by another method before being inputted into this function
    comp: This is a user defined number of components that the PCA should be applied with, this is a numerical input
    '''

    pca = PCA(n_components=comp)
    pca_data = pca.fit_transform(scaled_df)
    sns.pairplot(pd.DataFrame(pca_data))
    plt.show()

def plot_tsne(df,verb):

    '''
    This is a function that displays the tsne data of the dataframe, it takes the dataframe and takes a sample of it then plots the pairplot of the tsne data
    
    Args:
    df: The dataframe which holds all the data that needs to be plotted
    verb: The verbose value that the user defines, this is a numerical input 
    '''

    sample = df.sample(frac=0.4, random_state=1)
    tsne = TSNE(n_components=3, verbose=verb, random_state=1)
    tsne_data = tsne.fit_transform(sample)
    sns.pairplot(pd.DataFrame(tsne_data))
    plt.show()
    
def kmeans_range_clustering(lower_lim,upper_lim,df):

    '''
    This is a function that conducts clustering on the data with the use of the kmeans method for a range of k values defined by the user
    
    Args:
    lower_lim: The lower limit of the range of k values that the user defines, should be a numerical value
    upper_lim: The upper limit of the range of k values that the user defines, should be a numerical value
    '''

    ks = np.arange(lower_lim, upper_lim)
    inertia_list = []
    silhouette_score_list = []

    for k in ks:
        my_kmeans = KMeans(n_clusters=k, random_state = 1)
        
        y_labels = my_kmeans.fit_predict(df)
        
        inertia_list.append(my_kmeans.inertia_)

        silhouette = silhouette_score(df, y_labels)
        silhouette_score_list.append(silhouette)

        print(f"Computed Score for k={k}")
    
    plt.figure(figsize = (18,10))
    plt.plot(ks, inertia_list, marker='o')
    plt.xlabel('K - number of clusters')
    plt.ylabel('Inertia')
    plt.xticks(ks)
    plt.show()

    plt.figure(figsize = (18,10))
    plt.plot(ks, silhouette_score_list, marker='o')
    plt.xlabel('K - number of clusters')
    plt.ylabel('Silhouette score')
    plt.xticks(ks)
    plt.show()

def final_kmeans(comp, df):

    '''
    This function creates the final kmeans model after condcuting the exploration steps
    
    Args:
    df: This is the dataframe where the data is held for the kmeans clustering to occur
    comp: This is the number of cluster that is desired by the user, should be a numerical value
    '''


    kmeans = KMeans(n_clusters=comp)
    kmeans_y_labels = kmeans.fit_predict(df)

    df_kmeans = df.copy()
    df_kmeans['kmeans_labels'] = kmeans_y_labels

    sample = df_kmeans.sample(frac=0.4, random_state=1)

    tsne = TSNE(n_components=3, verbose=2, random_state=1)

    tsne_data = tsne.fit_transform(sample.drop('kmeans_labels', axis=1)) 

    tsne_df = pd.DataFrame(tsne_data, columns=[f'tSNE D{i + 1}' for i in range(tsne_data.shape[1])])
    tsne_df['kmeans_labels'] = sample['kmeans_labels'].values
    display(tsne_df)

    plt.figure(figsize = (18,10))
    sns.pairplot(tsne_df, hue="kmeans_labels", plot_kws={'alpha': 0.5})
    plt.show()

def plot_kmean_relative_means(comp, df):
    
    '''
    This is a function that plots the cluster groups of the kmeans model relative to the dataframe, also creates new features which are the groups for each 
    customer and returns it in the form of a final dataframe.
    
    Args:
    df: This is the dataframe where the data is held for the kmeans clustering to occur
    comp: This is the number of cluster that is desired by the user, should be a numerical value
    '''


    kmeans_final = KMeans(n_clusters=comp)
    
    kmeans_y_labels = kmeans_final.fit_predict(df)

    df_final = df.copy()
    df_final['kmeans_labels'] = kmeans_y_labels
    df_final.head()

    relative_means = df_final.groupby('kmeans_labels').mean()
    melted_relative_means_kmeans = relative_means.reset_index().melt(id_vars='kmeans_labels')

    melted_relative_means_kmeans = melted_relative_means_kmeans.sort_values("value")

    melted_relative_means_kmeans.head()

    f, ax = plt.subplots(figsize=(15,25))

    sns.barplot(x="value", y="variable", hue='kmeans_labels',data=melted_relative_means_kmeans)
    plt.xlim(-1.1, 1.1)
    plt.show()

    return df_final

def dendogram_clus(df):

    '''
    This is a function that plots the cluster groups in the form of a dendogram to help visualise the clusters better
    
    Args:
    df: This is the dataframe where the data is held for the clustering to occur
    '''

    linkage_mat = linkage(df, 'ward')
    plt.figure(figsize=(15,10))
    dendrogram(linkage_mat)
    plt.xticks([])
    plt.show()

def agglo_range_clust(lower_lim,upper_lim,df):

    '''
    This is a function that conducts agglomerative clustering on the data for a range of k values defined by the user
    
    Args:
    lower_lim: The lower limit of the range of k values that the user defines, should be a numerical value
    upper_lim: The upper limit of the range of k values that the user defines, should be a numerical value
    '''


    ks = np.arange(lower_lim, upper_lim)
    silhouette_list = []

    for k in ks:
        my_hclust = AgglomerativeClustering(n_clusters=k, linkage='ward')
        
        y_labels = my_hclust.fit_predict(df)
        
        silhouette = silhouette_score(df, y_labels)
        silhouette_list.append(silhouette)
        print(f"Computed Score for k={k}")
    
    plt.figure()
    plt.plot(ks, silhouette_list, marker='o')
    plt.xlabel('K - number of clusters')
    plt.ylabel('silhouette score')
    plt.xticks(ks)
    plt.show()

def final_agglo_custering(comp,df):

    '''
    This function creates a the final agglomerative model after exploration
    
    Args:
    df: This is the dataframe where the data is held for the agglomerative clustering to occur
    comp: This is the number of cluster that is desired by the user, should be a numerical value
    '''

    agg_model = AgglomerativeClustering(n_clusters=comp, linkage='ward')
    agg_y_labels = agg_model.fit_predict(df)


    cc_df_agglomerative = df.copy()
    cc_df_agglomerative['agglomerative_labels'] = agg_y_labels


    sample = cc_df_agglomerative.sample(frac=0.4, random_state=1)

    tsne = TSNE(n_components=3, verbose=2, random_state=1)

    # We need to drop the labels so tSNE won't use them when computing distances
    tsne_data = tsne.fit_transform(sample.drop('agglomerative_labels', axis=1)) 

    tsne_df = pd.DataFrame(tsne_data, columns=[f'tSNE D{i+1}' for i in range(tsne_data.shape[1])])
    tsne_df['agglomerative_labels'] = sample['agglomerative_labels'].values.astype(str)
    display(tsne_df)

    sns.pairplot(tsne_df, hue="agglomerative_labels", plot_kws={'alpha': 0.5})
    plt.show()


"""Modelling functions"""
def balance_data(X_train,y_train):
    '''
    This function balances the dataframe to account for class imbalances, this is only done on the train data and returns the balanced train data
    
    Args:
    X_train: The train data containing all the independent variables
    y_train: The train data for just the dependent variable
    '''

    X_train_sm, y_train_sm = SMOTE(random_state=1).fit_resample(X_train, y_train)

    print('Original train target variable distribution')
    display(pd.Series(y_train).value_counts().sort_index())

    print('\nResampled train target variable distribution')
    display(pd.Series(y_train_sm).value_counts().sort_index())

    return X_train_sm, y_train_sm

def plot_log_coeff(model,X_train):

    '''
    This function plots the coefficients of the logistic model to help visualise which features are most important
    
    Args:
    model: This the output of the logistic model after being ran
    X_train:The train data containing all the independent variables
    '''
    if type(model) == sklearn.model_selection._search.GridSearchCV:
        coeffs = pd.DataFrame(model.coef_[0], index = X_train.columns)
        coeffs = coeffs.T
        coeffs.rename(index={0:'Coefficient Value'},inplace=True)
        coeffs = coeffs.T
        coeffs.sort_values(by=['Coefficient Value'], ascending= False).plot(kind = 'bar',figsize = (25,7))
        plt.ylabel("Coefficient Value")
        plt.xticks(rotation = 45)
        plt.xlabel("Columns")
        plt.show()
    else:
        coeffs = pd.DataFrame(model.coef_[0], index = X_train.columns)
        coeffs = coeffs.T
        coeffs.rename(index={0:'Coefficient Value'},inplace=True)
        coeffs = coeffs.T
        coeffs.sort_values(by=['Coefficient Value'], ascending= False).plot(kind = 'bar',figsize = (25,7))
        plt.ylabel("Coefficient Value")
        plt.xticks(rotation = 45)
        plt.xlabel("Columns")
        plt.show()

def plot_model_result(model_type, X_train, X_test, y_train, y_test):
    
    '''This function plots a graph for the most optimised hyperparameter for each specific model chosen, logistic, 
    decision tree, knearest neighbours etc.

    A range for each models specific hyperparameters is predetermined and the model then plots how well it performs on the test
    and train data. It then returns the final ran model for further optimisation if need be.
    
    Args:
    X_train: The train data containing all the independent variables
    y_train: The train data for just the dependent variable
    X_test: The test data containing all the independent variables
    y_test: The test data for just the dependent variable
    
    '''

    train_acc = []
    test_acc = []

    if model_type == "logistic":
        
        co = [0.001, 0.01, 0.1, 1, 10, 100, 1000, 10000]
        
        for param in co:
            logreg = LogisticRegression(C = param,solver = 'lbfgs',max_iter = 10000)
            logreg.fit(X_train, y_train)

            train_acc.append(logreg.score(X_train, y_train))
            test_acc.append(logreg.score(X_test, y_test))

        plt.figure(figsize=(10,6))
        plt.plot(co,train_acc, label = 'Train')
        plt.plot(co,test_acc, label = 'Test')
        plt.legend()
        plt.xscale('log')
        plt.ylabel("Accuracy Score")
        plt.xlabel("C Value")
        plt.show()

        index_val = test_acc.index(max(test_acc))
        max_val = co[index_val]
        logreg = LogisticRegression(C = max_val, max_iter = 10000)
        logreg.fit(X_train, y_train)
        
        print("The C value which yielded the highest test accuracy was: "+str(max_val)+", I will input this in my final model:")
        print(f"Train score: {logreg.score(X_train, y_train)}")
        print(f"Test score: {logreg.score(X_test, y_test)}")

        plot_confusion_matrix(logreg,X_test,y_test)
        plt.show()
        
        y_test_pred = logreg.predict(X_test)

        print(classification_report(y_test,y_test_pred))
        
        return logreg 
        
    
    if model_type == "decision":
        
        dep = [i for i in range(1,10)]

        for depth in dep:
            
            dt = DecisionTreeClassifier(max_depth = depth)
            dt_fitted = dt.fit(X_train, y_train)

            train_acc.append(dt_fitted.score(X_train, y_train))
            test_acc.append(dt_fitted.score(X_test, y_test))

        plt.figure(figsize=(10,6))
        plt.plot(dep, train_acc, color = 'purple', label = 'train')
        plt.plot(dep, test_acc, color = 'green', label = 'test')
        plt.ylabel("Accuracy Score")
        plt.xlabel("Max_Depth Value")
        plt.legend()
        plt.show()

        index_val = test_acc.index(max(test_acc))
        max_val = dep[index_val]
        
        dt = DecisionTreeClassifier(max_depth = max_val)
        dt_fitted = dt.fit(X_train, y_train)
        
        print("The max depth value which yielded the highest test accuracy was: "+str(max_val)+", I will input this in my final model:")
        print(f"Train score: {dt_fitted.score(X_train, y_train)}")
        print(f"Test score: {dt_fitted.score(X_test, y_test)}")

        plot_confusion_matrix(dt_fitted,X_test,y_test)
        plt.show()

        y_test_pred = dt_fitted.predict(X_test)

        print(classification_report(y_test,y_test_pred))
        
        return dt_fitted


    if model_type == "knearest":
        
        neighbours = [k for k in range(1,50,2)]

        for k in neighbours:
            KNN_model = KNeighborsClassifier(n_neighbors=k)
            KNN_model.fit(X_train, y_train)
            train_acc.append(KNN_model.score(X_train, y_train))
            test_acc.append(KNN_model.score(X_test, y_test))

        plt.figure(figsize=(10,6))
        plt.plot(neighbours,train_acc, label = 'Train')
        plt.plot(neighbours,test_acc, label = 'Test')
        plt.ylabel("Accuracy Score")
        plt.xlabel("Neighbours value")
        plt.legend()
        plt.show()

        index_val = test_acc.index(max(test_acc))
        max_val = neighbours[index_val]
        KNN_model = KNeighborsClassifier(n_neighbors= max_val)
        KNN_model.fit(X_train, y_train)

        
        print("The neighbours value which yielded the highest test accuracy was: "+str(max_val)+", I will input this in my final model:")
        print(f"Train score: {KNN_model.score(X_train, y_train)}")
        print(f"Test score: {KNN_model.score(X_test, y_test)}")

        plot_confusion_matrix(KNN_model,X_test,y_test)
        plt.show()
        
        y_test_pred = KNN_model.predict(X_test)

        print(classification_report(y_test,y_test_pred))

        return KNN_model

    if model_type=='randomforest':
        
        estimators = [i for i in range(1,100)]

        for i in estimators:
            my_random_forest = RandomForestClassifier(n_estimators=i)
            my_random_forest.fit(X_train, y_train)

            train_acc.append(my_random_forest.score(X_train, y_train))
            test_acc.append(my_random_forest.score(X_test, y_test))

        plt.figure(figsize=(10,6))
        plt.plot(estimators,train_acc, label = 'Train')
        plt.plot(estimators,test_acc, label = 'Test')
        plt.ylabel("Accuracy Score")
        plt.xlabel("Number of estimators")
        plt.legend()
        plt.show()

        index_val = test_acc.index(max(test_acc))
        max_val = estimators[index_val]
        my_random_forest = RandomForestClassifier(n_estimators=max_val)
        my_random_forest.fit(X_train, y_train)

        
        print("The number of trees which yielded the highest test accuracy was: "+str(max_val)+", I will input this in my final model:")
        print(f"Train score: {my_random_forest.score(X_train, y_train)}")
        print(f"Test score: {my_random_forest.score(X_test, y_test)}")

        plot_confusion_matrix(my_random_forest,X_test,y_test)
        plt.show()
        
        y_test_pred = my_random_forest.predict(X_test)

        print(classification_report(y_test,y_test_pred))

        return my_random_forest

    if model_type=='adaboost':
        
        estimators = [i for i in range(1,100)]

        for i in estimators:
            AdaBoost_model = AdaBoostClassifier(n_estimators=i)
            AdaBoost_model.fit(X_train, y_train)

            train_acc.append(AdaBoost_model.score(X_train, y_train))
            test_acc.append(AdaBoost_model.score(X_test, y_test))

        plt.figure(figsize=(10,6))
        plt.plot(estimators,train_acc, label = 'Train')
        plt.plot(estimators,test_acc, label = 'Test')
        plt.ylabel("Accuracy Score")
        plt.xlabel("Number of estimators")
        plt.legend()
        plt.show()

        index_val = test_acc.index(max(test_acc))
        max_val = estimators[index_val]
        AdaBoost_model = AdaBoostClassifier(n_estimators=max_val)
        AdaBoost_model.fit(X_train, y_train)

        
        print("The number of trees which yielded the highest test accuracy was: "+str(max_val)+", I will input this in my final model:")
        print(f"Train score: {AdaBoost_model.score(X_train, y_train)}")
        print(f"Test score: {AdaBoost_model.score(X_test, y_test)}")

        plot_confusion_matrix(AdaBoost_model,X_test,y_test)
        plt.show()
        
        y_test_pred = AdaBoost_model.predict(X_test)

        print(classification_report(y_test,y_test_pred))

        return AdaBoost_model

    if model_type=='xgboost':
        
        estimators = [i for i in range(1,100)]

        for i in estimators:
            XGB_model = XGBClassifier(n_estimators=i, verbosity = 0)
            XGB_model.fit(X_train, y_train)

            train_acc.append(XGB_model.score(X_train, y_train))
            test_acc.append(XGB_model.score(X_test, y_test))

        plt.figure(figsize=(10,6))
        plt.plot(estimators,train_acc, label = 'Train')
        plt.plot(estimators,test_acc, label = 'Test')
        plt.ylabel("Accuracy Score")
        plt.xlabel("Number of estimators")
        plt.legend()
        plt.show()

        index_val = test_acc.index(max(test_acc))
        max_val = estimators[index_val]
        XGB_model = XGBClassifier(n_estimators=max_val)
        XGB_model.fit(X_train, y_train)

        
        print("The number of trees which yielded the highest test accuracy was: "+str(max_val)+", I will input this in my final model:")
        print(f"Train score: {XGB_model.score(X_train, y_train)}")
        print(f"Test score: {XGB_model.score(X_test, y_test)}")

        plot_confusion_matrix(XGB_model,X_test,y_test)
        plt.show()
        
        y_test_pred = XGB_model.predict(X_test)

        print(classification_report(y_test,y_test_pred))

        return XGB_model        


def plot_model_coefficients(model, X_train):




    words_count = pd.DataFrame(model.coef_[0], index = X_train.columns)
    words_count = words_count.T
    words_count.rename(index={0:'Coefficient Value'},inplace=True)
    words_count = words_count.T
    words_count.sort_values(by=['Coefficient Value'], ascending= False).plot(kind = 'bar',figsize = (25,7))
    plt.ylabel("Coefficient Value")
    plt.xticks(rotation = 45)
    plt.xlabel("Columns")
    plt.show()

def display_confusion_matrix(model, X_test,y_test):

    '''
    This function plots the confusion matrix for the final output of the model as well as a classification report
    
    Args:
    model: The final model after being run 
    X_test:The test data containing all the independent variables
    y_test: The test data containing all the dependent variables
    '''
    
    plot_confusion_matrix(model,X_test,y_test)
    plt.show()
        
    y_test_pred = model.predict(X_test)

    print(classification_report(y_test,y_test_pred))



def print_scores(model_name,X_test,y_test):
    '''
    This function prints out the scores for the model, on the test data and the parameters for the best estimator

    Arg:
    model: This is the machine learning model that I want to see the scores and estimators for 
    X_test: The test data containing all the independent variables
    y_test: The test data for just the dependent variable
    '''


    print("The parameters for the best estimator is: ")
    print(model_name.best_estimator_[-1].get_params())
    print("The score for the test data :")
    print(model_name.score(X_test,y_test))
    print("The best score from the cross validation fitting :")
    print(model_name.best_score_)

def decision_tree_feature_importance(model_name,X_train):
    '''
    This model takes the importance of all the features in the decision tree model and produces an importance value and a normalised importance value
    The output will then be displayed as a bar chart

    Args:
    model_name: This is the final decision tree model with the features displayed in the form of a bar chart 
    X_train: The train data of the independent variables
    '''

    importances_df = pd.DataFrame({'Variable': X_train.columns,'Importance': model_name.tree_.compute_feature_importances(normalize=False),'Normalized Importance': model_name.feature_importances_})
    importances_df.sort_values(by='Importance', ascending=False, inplace=True, ignore_index=True)
    importances_df = importances_df.head(10)

    plt.figure(figsize=(25,10))
    plt.bar(importances_df['Variable'], importances_df['Normalized Importance'], color = 'orange')
    plt.title('Decision Tree Feature Importance by MDI', fontsize=16)
    plt.ylabel('Importance (%)')
    plt.xlabel('Feature')
    plt.xticks(rotation = 45)
    sns.despine()
    plt.show()


def ensemble_tree_feature_importance(model_name,X_train):

    '''
    This function displays the feature importance for ensemble tree methods in the form of a bar plot
    Args:
    model_name: The final optimised model
    X_train: The train data of the independent variables
    '''

    importances = model_name.feature_importances_
    std = np.std([tree.feature_importances_ for tree in model_name.estimators_], axis=0)
    forest_importances = pd.DataFrame({'Variable': X_train.columns,'Importance': importances })
    forest_importances.sort_values(by='Importance', ascending=False, inplace=True, ignore_index=True)
    forest_importances = forest_importances.head(10)

    plt.figure(figsize=(25,10))
    plt.bar(forest_importances['Variable'], forest_importances['Importance'], color = 'orange')
    plt.title('Tree Feature Importance by MDI', fontsize=16)
    plt.ylabel('Importance ')
    plt.xlabel('Feature')
    plt.xticks(rotation = 45)
    sns.despine()
    plt.show()

def xgboost_feature_importance(X_train,model_name):

    '''
    This function plots the feature importance of an xgboost as long as the booster is a tree
    Args:
    X_train: The train data of the independent variables
    model_name: The final optimised model
    '''

    cols = [X_train.columns[i] for i in range(0,len(X_train.columns))]
    cols = model_name.get_booster().feature_names
    feature_important = model_name.get_booster().get_fscore()
    xgb_features = pd.DataFrame.from_dict(feature_important,orient='index',columns=['Importance'])
    xgb_features.sort_values("Importance", ascending = False).head(10).plot(kind = 'bar',color='orange',figsize = (15,10))
    plt.title("Bar plot demonstrating feature importance in XGBoost model")
    plt.xlabel('Features')
    plt.ylabel('Importance')
    plt.xticks(rotation = 55)
    plt.show()


