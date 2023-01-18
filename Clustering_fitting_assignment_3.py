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
# Defind a fuction to read the files

def read_data(file_name, country, save_file): 
    df = pd.read_excel(file_name, header = [3]) # To read the file
    df = df[df['Country Name'].isin([country])].reset_index(drop = True)
    df = df.T.reset_index(drop = False)
    new_col1 = 'Year'
    new_col2 = df.iloc[2,1]
    df.columns = [new_col1, new_col2] # Set the columns name
    df = df.iloc[4:,:].reset_index(drop = True)
    df = df.dropna().reset_index(drop = True) #To drop the column
    df = df.astype(float)
    print(df) # To print the data
    df.to_excel(save_file) #To save the file 
    return df

df1 = read_data('GDP PER CAPITA.xls', 'India', 'GDP_INDIA.xlsx')  # Calling the function to read the file
df2 = read_data('Adjusted net national income per capita.xls', 'India', 'Income_India.xlsx')  # Calling the function to read the file
df3 = read_data('GDP PER CAPITA.xls', 'Singapore', 'GDP_Singapore.xlsx')  # Calling the function to read the file
df4 = read_data('Adjusted net national income per capita.xls', 'Singapore', 'Income_Singapore.xlsx')  # Calling the function to read the file
df5 = read_data('GDP PER CAPITA.xls', 'Italy', 'GDP_ITALY.xlsx')  # Calling the function to read the file
df6 = read_data('Adjusted net national income per capita.xls', 'Italy', 'Income_Italy.xlsx')  # Calling the function to read the file

def exp_growth(t, scale, growth):
    f = scale * np.exp(growth * (t-1960)) 
    return f
       

# fitting of exponential function to df-1
param, covar = opt.curve_fit(exp_growth, df1['Year'], df1['GDP per capita (current US$)'], p0=(2e9, 0.05), maxfev = 2000)
df1['fit'] = exp_growth(df1['Year'], *param)

plt.figure(dpi = 300) #To create the figure    
plt.plot(df1['Year'], df1['GDP per capita (current US$)'], label = 'GDP per capita (current US$)') #To plot the graph
plt.xlabel('Year') #Providing x label
plt.ylabel('GDP per capita (current US$)') #Providing the y label
plt.legend() #Providing the legend
plt.show() #To show the plot

sigma = np.sqrt(np.diag(covar)) #Defining the sigma function
print(sigma) #Printing the sigma
low, up = err.err_ranges(df1["Year"], exp_growth, param, sigma) #Low and up error range
print('lower limit:', low, 'upper limt:', up)
plt.figure(dpi = 300) #To create figure size
plt.plot(df1['Year'], df1['GDP per capita (current US$)'], label = 'GDP per capita (current US$)',color = 'black') #To create the plot
plt.plot(df1['Year'], df1['fit'], label = 'fit',color = 'red') #Plot the data
plt.fill_between(df1['Year'], low, up,color="green", alpha = 0.7) #Fill the upper and lower data
plt.title('Fitting with Exponential function(Country-India)') #To provide the title
plt.xlabel('Year') #To provide the xlabel
plt.ylabel('GDP per capita (current US$)') #To provide the ylabel
plt.legend(loc='upper left') #To show the legend
plt.show() # To show the plot

print("Forcasted GDP Per Capita of India")
low, up = err.err_ranges(2030, exp_growth, param, sigma) #low and up error range with exponential function
mean = (up+low) / 2.0
pm = (up-low) / 2.0
print("2030:", mean, "+/-", pm)
low, up = err.err_ranges(2040, exp_growth, param, sigma)
mean = (up+low) / 2.0
pm = (up-low) / 2.0
print("2040:", mean, "+/-", pm)
low, up = err.err_ranges(2050, exp_growth, param, sigma)
mean = (up+low) / 2.0
pm = (up-low) / 2.0
print("2050:", mean, "+/-", pm)

# fit exponential function to df-2
param, covar = opt.curve_fit(exp_growth, df2['Year'], df2['Adjusted net national income per capita (current US$)'], p0=(2e9, 0.05), maxfev = 2000)
df2['fit'] = exp_growth(df2['Year'], *param)
plt.figure(dpi = 300)    
plt.plot(df2['Year'], df2['Adjusted net national income per capita (current US$)'], label = 'Adjusted net national income per capita (current US$)')
plt.plot(df2['Year'], df2['fit'], label = 'fit') #To plot the fit line
plt.xlabel('Year') #To provide the xlabel
plt.ylabel('Adjusted net national income per capita (current US$)') #To provide the ylabel
plt.legend(loc='upper left') #To show the legend
plt.show() # To show the plot

sigma = np.sqrt(np.diag(covar)) #Defining the sigma function
print(sigma) #Print the sigma 
low, up = err.err_ranges(df2["Year"], exp_growth, param, sigma) #Low and up error range
#print('lower limit:', low, 'upper limt:', up)
plt.figure(dpi = 300) #To create the figure
plt.plot(df2['Year'], df2['Adjusted net national income per capita (current US$)'], label = 'National income per capita (current US$)',color = 'black') #To plot the data
plt.plot(df2['Year'], df2['fit'], label = 'fit',color = 'red') #To plot the fit line
plt.fill_between(df2['Year'], low, up,color="Purple", alpha = 0.7) #To fill the upper and lower error
plt.title('Fitting with Exponential function(Country-India)') #To provide the title
plt.xlabel('Year') #To provide the xlabel
plt.ylabel('National income per capita (current US$)') #To provide the ylabel
plt.legend(loc='upper left') #To show the legend
plt.show() # To show the plot

print("Forcasted national Income per capita of India")
low, up = err.err_ranges(2030, exp_growth, param, sigma)
mean = (up+low) / 2.0
pm = (up-low) / 2.0
print("2030:", mean, "+/-", pm)
low, up = err.err_ranges(2040, exp_growth, param, sigma)
mean = (up+low) / 2.0
pm = (up-low) / 2.0
print("2040:", mean, "+/-", pm)
low, up = err.err_ranges(2050, exp_growth, param, sigma)
mean = (up+low) / 2.0
pm = (up-low) / 2.0
print("2050:", mean, "+/-", pm)

# fit exponential function to df-3

param, covar = opt.curve_fit(exp_growth, df3['Year'], df3['GDP per capita (current US$)'], p0=(1700, 0.1), maxfev = 2000)
df3['fit'] = exp_growth(df3['Year'], *param)

plt.figure(dpi = 300)    
plt.plot(df3['Year'], df3['GDP per capita (current US$)'], label = 'GDP per capita (current US$)')
plt.plot(df3['Year'], df3['fit'], label = 'fit') #Plot the data
#plt.ylim(50, 65)
plt.xlabel('Year') #Providing x label
plt.ylabel('GDP per capita (current US$)') #Providing the y label
plt.legend() #Providing the legend
plt.show() #To show the plot

sigma = np.sqrt(np.diag(covar))
print(sigma)
low, up = err.err_ranges(df3["Year"], exp_growth, param, sigma)
#print('lower limit:', low, 'upper limt:', up)
plt.figure(dpi = 300)
plt.plot(df3['Year'], df3['GDP per capita (current US$)'], label = 'GDP per capita (current US$)',color = 'black')
plt.plot(df3['Year'], df3['fit'], label = 'fit',color = 'red') #Plot the data
plt.fill_between(df3['Year'], low, up,color="yellow", alpha = 0.7) #Fill the upper and lower data
plt.title('Fitting with Exponential function(Country-Singapore)') #To provide the title
plt.xlabel('Year') #To provide the xlabel
plt.ylabel('GDP per capita (current US$)') #To provide the ylabel
plt.legend(loc='upper left') #To show the legend
plt.show() # To show the plot

print("Forcasted GDP Per Capita of Singapor")
low, up = err.err_ranges(2030, exp_growth, param, sigma)
mean = (up+low) / 2.0
pm = (up-low) / 2.0
print("2030:", mean, "+/-", pm)
low, up = err.err_ranges(2040, exp_growth, param, sigma)
mean = (up+low) / 2.0
pm = (up-low) / 2.0
print("2040:", mean, "+/-", pm)
low, up = err.err_ranges(2050, exp_growth, param, sigma)
mean = (up+low) / 2.0
pm = (up-low) / 2.0
print("2050:", mean, "+/-", pm)

# fit exponential function to df-4
param, covar = opt.curve_fit(exp_growth, df4['Year'], df4['Adjusted net national income per capita (current US$)'], p0=(2e9, 0.05), maxfev = 2000)
df4['fit'] = exp_growth(df4['Year'], *param)
plt.figure(dpi = 300)    
plt.plot(df4['Year'], df4['Adjusted net national income per capita (current US$)'], label = 'Adjusted net national income per capita (current US$)')
plt.plot(df4['Year'], df4['fit'], label = 'fit') #To plot the fit line
plt.xlabel('Year') #To provide the xlabel
plt.ylabel('Adjusted net national income per capita (current US$)') #To provide the ylabel
plt.legend(loc='upper left') #To show the legend
plt.show() # To show the plot

sigma = np.sqrt(np.diag(covar))
print(sigma)
low, up = err.err_ranges(df4["Year"], exp_growth, param, sigma)
#print('lower limit:', low, 'upper limt:', up)
plt.figure(dpi = 300)
plt.plot(df4['Year'], df4['Adjusted net national income per capita (current US$)'], label = 'National income per capita (current US$)',color = 'black') #To plot the data
plt.plot(df4['Year'], df4['fit'], label = 'fit',color = 'red') #To plot the fit line
plt.fill_between(df4['Year'], low, up,color="blue", alpha = 0.7) #To fill the upper and lower error
plt.title('Fitting with Exponential function(Country-Singapore)') #To provide the title
plt.xlabel('Year') #To provide the xlabel
plt.ylabel('National income per capita (current US$)') #To provide the ylabel
plt.legend(loc='upper left') #To show the legend
plt.show() # To show the plot

print("Forcasted national Income per capita of Singapore")
low, up = err.err_ranges(2030, exp_growth, param, sigma)
mean = (up+low) / 2.0
pm = (up-low) / 2.0
print("2030:", mean, "+/-", pm)
low, up = err.err_ranges(2040, exp_growth, param, sigma)
mean = (up+low) / 2.0
pm = (up-low) / 2.0
print("2040:", mean, "+/-", pm)
low, up = err.err_ranges(2050, exp_growth, param, sigma)
mean = (up+low) / 2.0
pm = (up-low) / 2.0
print("2050:", mean, "+/-", pm)

# fit exponential function to df-5

param, covar = opt.curve_fit(exp_growth, df5['Year'], df5['GDP per capita (current US$)'], p0=(1700, 0.1), maxfev = 2000)
df5['fit'] = exp_growth(df5['Year'], *param)

plt.figure(dpi = 300)    
plt.plot(df5['Year'], df5['GDP per capita (current US$)'], label = 'GDP per capita (current US$)')
plt.plot(df5['Year'], df5['fit'], label = 'fit') #Plot the data
#plt.ylim(50, 65)
plt.xlabel('Year') #Providing x label
plt.ylabel('GDP per capita (current US$)') #Providing the y label
plt.legend() #Providing the legend
plt.show() #To show the plot

sigma = np.sqrt(np.diag(covar))
print(sigma)
low, up = err.err_ranges(df5["Year"], exp_growth, param, sigma)
#print('lower limit:', low, 'upper limt:', up)
plt.figure(dpi = 300)
plt.plot(df5['Year'], df5['GDP per capita (current US$)'], label = 'GDP per capita (current US$)',color = 'black')
plt.plot(df5['Year'], df5['fit'], label = 'fit',color = 'red') #Plot the data
plt.fill_between(df5['Year'], low, up,color="maroon", alpha = 0.7) #Fill the upper and lower data
plt.title('Fitting with Exponential function(Country-Italy)') #To provide the title
plt.xlabel('Year') #To provide the xlabel
plt.ylabel('GDP per capita (current US$)') #To provide the ylabel
plt.legend(loc='upper left') #To show the legend
plt.show() # To show the plot

print("Forcasted GDP Per Capita of Italy")
low, up = err.err_ranges(2030, exp_growth, param, sigma)
mean = (up+low) / 2.0
pm = (up-low) / 2.0
print("2030:", mean, "+/-", pm)
low, up = err.err_ranges(2040, exp_growth, param, sigma)
mean = (up+low) / 2.0
pm = (up-low) / 2.0
print("2040:", mean, "+/-", pm)
low, up = err.err_ranges(2050, exp_growth, param, sigma)
mean = (up+low) / 2.0
pm = (up-low) / 2.0
print("2050:", mean, "+/-", pm)

# fit exponential function to df-6
param, covar = opt.curve_fit(exp_growth, df6['Year'], df6['Adjusted net national income per capita (current US$)'], p0=(1880,0.01), maxfev = 2000)
df6['fit'] = exp_growth(df6['Year'], *param)
plt.figure(dpi = 300) #To create the figure size    
plt.plot(df6['Year'], df6['Adjusted net national income per capita (current US$)'], label = 'Adjusted net national income per capita (current US$)')
plt.plot(df6['Year'], df6['fit'], label = 'fit') #To plot the fit line
plt.xlabel('Year') #To provide the xlabel
plt.ylabel('Adjusted net national income per capita (current US$)') #To provide the ylabel
plt.legend(loc='upper left') #To show the legend
plt.show() # To show the plot

sigma = np.sqrt(np.diag(covar))
print(sigma)
low, up = err.err_ranges(df6["Year"], exp_growth, param, sigma)
#print('lower limit:', low, 'upper limt:', up)
plt.figure(dpi = 300)
plt.plot(df6['Year'], df6['Adjusted net national income per capita (current US$)'], label = 'National income per capita (current US$)',color = 'black') #To plot the data
plt.plot(df6['Year'], df6['fit'], label = 'fit',color = 'red') #To plot the fit line
plt.fill_between(df6['Year'], low, up,color="pink", alpha = 0.7) #To fill the upper and lower error
plt.title('Fitting with Exponential function(Country-Italy)') #To provide the title
plt.xlabel('Year') #To provide the xlabel
plt.ylabel('National income per capita (current US$)') #To provide the ylabel
plt.legend(loc='upper left') #To show the legend
plt.show() # To show the plot

print("Forcasted national Income per capita of Italy")
low, up = err.err_ranges(2030, exp_growth, param, sigma)
mean = (up+low) / 2.0
pm = (up-low) / 2.0
print("2030:", mean, "+/-", pm)
low, up = err.err_ranges(2040, exp_growth, param, sigma)
mean = (up+low) / 2.0
pm = (up-low) / 2.0
print("2040:", mean, "+/-", pm)
low, up = err.err_ranges(2050, exp_growth, param, sigma)
mean = (up+low) / 2.0
pm = (up-low) / 2.0
print("2050:", mean, "+/-", pm)
