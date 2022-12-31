# Importing All the Python Inbuit Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
import sklearn.cluster as cluster
import sklearn.metrics as skmet 
import scipy.optimize as opt
import err_ranges as err
from matplotlib.legend_handler import HandlerTuple

def Read(file): #Define the function to read the excel file
    dataframe=pd.read_excel(file,header=[3]) #Function to read the file
    dataframe = dataframe.iloc[:,[0,63]]
    print(dataframe) #Print the dataframe
    return dataframe #Return the function

df_GDP = Read("GDP PER CAPITA.xls") #Calling the function with argument to read file
df_GDP.rename(columns = {'2019' : 'GDP PER CAPITA in 2019'}, inplace = True) #Rename the column Name
print(df_GDP) #Print the dataframe

df_Exports_services = Read("Exports of goods and services.xls") #Calling the function with argument to read file
df_Exports_services.rename(columns = {'2019' : 'Exports of goods and services in 2019'}, inplace = True) #Rename the column Name
print(df_Exports_services) #Print the dataframe

df_expence = Read("expence.xls") #Calling the function with argument to read file
df_expence.rename(columns = {'2019' :'expence in 2019'}, inplace = True) #Rename the column Name
print(df_expence) #Print the dataframe

df_Current_health_expenditure = Read("Current health expenditure.xls") #Calling the function with argument to read file
df_Current_health_expenditure.rename(columns = {'2019' :'Current health expenditure in 2019'}, inplace = True) #Rename the column Name
print(df_expence) #Print the dataframe

df_income = Read("Adjusted net national income per capita.xls") #Calling the function with argument to read file
df_income.rename(columns = {'2019' :'Adjusted net national income per capita in 2019'}, inplace = True) #Rename the column Name
print(df_income) #Print the dataframe

df_Life_expectancy = Read("Life expectancy at birth.xls") #Calling the function with argument to read file
df_Life_expectancy.rename(columns = {'2019' : 'Life expectancy at birth in 2019'}, inplace = True) #Rename the column Name
print(df_Life_expectancy) #Print the dataframe
#merging the dataframe
df_merge = pd.merge(df_Exports_services,df_GDP, on = 'Country Name').merge(df_expence,on = 'Country Name').merge(df_Current_health_expenditure,on = 'Country Name').merge(df_income,on = 'Country Name').merge(df_Life_expectancy,on = 'Country Name')
print(df_merge) #Print the dataframe
df_merge.to_excel('df_merge.xlsx') #Save the dataframe in to excelfile
pd.plotting.scatter_matrix(df_merge,s=120,c="red", figsize=(15.0, 15.0)) #Plot the scatter plot
plt.suptitle('Scatter matrix of Indicators',size=25) #Providing the title
plt.tight_layout() # helps to avoid overlap of abels
plt.show() #To show the plot

def norm(array): # Define functions to normalise one array and iterate over all numerical columns of the dataframe
    #Returns array normalised to [0,1]. Array can be a numpy array or a column of a dataframe"""
    min_val = np.min(array) 
    max_val = np.max(array)
    scaled = (array-min_val) / (max_val-min_val)  
    return scaled

def norm_df(df, first=0, last=None): #Creating a function to make column of the dataframe to  normalised  [0,1]
 
    # iterate over all numerical columns
    for col in df.columns[first:last]:     # excluding the first column
        df[col] = norm(df[col])
    return df

# Read in and inspect # reading the file and basic statistics
df_cluster = pd.merge(df_GDP,df_income, on = 'Country Name') #Making a new dataframe
df_cluster = df_cluster.dropna().reset_index(drop = True) #Droping the NA value
features = ['GDP PER CAPITA in 2019', 'Adjusted net national income per capita in 2019'] #extracting the required column
clusterdata = df_cluster[features].copy() #Making a new dataframe for clustering
print(clusterdata.describe()) #Printing the statistical data
# extract columns for fitting
df_fit = clusterdata[["GDP PER CAPITA in 2019", "Adjusted net national income per capita in 2019"]].copy()
df_fit = norm_df(df_fit) #normalisation is done only on the extract columns. .copy() prevents
print(df_fit.describe()) #Printing the statistical data

for ic in range(2, 7):
    # set up kmeans and fit
    kmeans = cluster.KMeans(n_clusters=ic)
    kmeans.fit(df_fit)     

    # extract labels and calculate silhoutte score
    labels = kmeans.labels_
    print (ic, skmet.silhouette_score(df_fit, labels))

# Plot for two clusters
kmeans = cluster.KMeans(n_clusters=2,random_state = 0)
xyz = kmeans.fit(df_fit)     
labels = kmeans.labels_ # extract labels and cluster centres
cen = kmeans.cluster_centers_ #To create centers
relabel = np.choose(labels,[0,1]).astype(np.int64) #For relabel
colors = np.array(["Red","Green"]) #define the colors
plt.figure(figsize=(6.0, 6.0)) #Creating the figure size
# Individual colours can be assigned to symbols. The label l is used to the select the 
# l-th number from the colour table.
scatter = plt.scatter(df_fit["GDP PER CAPITA in 2019"], df_fit["Adjusted net national income per capita in 2019"], c = colors[relabel])

# show cluster centres
for ic in range(2):
    xc, yc = cen[ic,:]
    plt.plot(xc, yc, "dk", markersize=10)
    
plt.xlabel("GDP PER CAPITA in 2019") #Providing the xlabel
plt.ylabel("Adjusted net national income per capita in 2019") #Providing the ylabel
plt.title("2 clusters") #Providing the title
plt.legend(handles=scatter.legend_elements()[0], title="Countries") #Providing the title
plt.show() #To show the plot

# Plot for three clusters
kmeans = cluster.KMeans(n_clusters=3) #Define the kmean
xyz = kmeans.fit(df_fit)     

# extract labels and cluster centres
labels = kmeans.labels_ #Define the label
cen = kmeans.cluster_centers_
relabel = np.choose(labels,[0,1,2]).astype(np.int64)
colors = np.array(["Red","Green","Blue"])
plt.figure(figsize=(6.0, 6.0)) #To decide the figure size
# Individual colours can be assigned to symbols. The label l is used to the select the 
# l-th number from the colour table.
plt.scatter(df_fit["GDP PER CAPITA in 2019"], df_fit["Adjusted net national income per capita in 2019"], c = colors[relabel])

# show cluster centres
for ic in range(3):
    xc, yc = cen[ic,:]
    plt.plot(xc, yc, "dk", markersize=10)
    
plt.xlabel("GDP PER CAPITA in 2019") #Providing the xlabel
plt.ylabel("Adjusted net national income per capita in 2019") #Providing the ylabel
plt.title("3 clusters") #Providing the title
plt.legend(handles=scatter.legend_elements()[0], title="Countries") #Providing the title
plt.show() #Show the plot
cluster1 = df_cluster[labels == 0].reset_index(drop = True) #Providing the label
cluster2 = df_cluster[labels == 1].reset_index(drop = True) #Providing the label
cluster3 = df_cluster[labels == 2].reset_index(drop = True) #Providing the label
cluster1.to_excel('country_group1.xlsx') #Creating the new excelfile
cluster2.to_excel('country_group2.xlsx') #Creating the new excelfile
cluster3.to_excel('country_group3.xlsx') #Creating the new excelfile
