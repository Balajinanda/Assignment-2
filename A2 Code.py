# -*- coding: utf-8 -*-
"""
Created on Wed Apr  5 22:56:50 2023

@author: Balaj
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as stats

#Load data from World Bank data
df = pd.read_csv(r"C:\Users\Balaj\OneDrive\Desktop\Ass - 2\Data.csv", \
                 skiprows=4)
print(df.head())

#Describe the given data
df.describe()

#Data information
df.info()

#Creating the function that will take the file name as an argument.


def co2_emissions_data(filename):
    """
    Read a CSV file containing CO2 emissions data in Worldbank format and \
        return two dataframes: 
    one with years as columns and one with countries as columns.
    
    Args:
    - filename: string, the name of the CSV file to read
    
    Returns:
    - years_df: pandas dataframe, a dataframe with years as columns and \
        CO2 emissions data for each country
    - countries_df: pandas dataframe, a dataframe with countries as \
        columns and CO2 emissions data for each year
    """

    # Extract the data for the years 1995-2005
    years_df = df.loc[:, 'Country Name':'2005']
    years_df.columns = [col if not col.isdigit() else str(col) for\
                        col in years_df.columns]
    
    # Transpose the DataFrame to get a country-centric view
    countries_df = years_df.transpose()
    
    # Replace empty values with 0
    countries_df = countries_df.fillna(0)
    
    # Set the column names for the countries DataFrame
    countries_df.columns = countries_df.iloc[0]
    countries_df = countries_df.iloc[1:]
    countries_df.index.name = 'Year'
    
    # Set the column names for the years DataFrame
    years_df = years_df.rename(columns={'Country Name': 'Year'})
    years_df = years_df.set_index('Year')
    
    return years_df, countries_df


#calling the function we created above
years_df, countries_df = co2_emissions_data("Data.csv")
print(years_df)
print(countries_df)

#Describe years data
years_df.describe()


def plot_grouped_bar_graph(df, countries, years, indicator):
    """
    Plots a grouped bar graph of the given indicator for the\
        specified countries and years.
    """
    # Filter the DataFrame for the selected countries and indicator
    df_filtered = df[(df['Country Name'].isin(countries)) & \
                     (df['Indicator Name'] == indicator)]
    df_filtered = df_filtered[['Country Name'] + years]

    # Group data by country and calculate sum of values for each year
    df_grouped = df_filtered.groupby('Country Name')[years].sum()

    # Reshape data for grouped bar graph
    df_melted = pd.melt(df_grouped.reset_index(), id_vars='Country Name', \
                        value_vars=years)
    df_melted.columns = ['Country', 'Year', indicator]

    # Pivot data for grouped bar graph
    df_pivot = df_melted.pivot(index='Year', columns='Country', \
                               values=indicator)

    # Transpose DataFrame to use years as columns
    df_pivot = df_pivot.T

    # Plot grouped bar graph
    fig, ax = plt.subplots(figsize=(10,6))
    df_pivot.plot(kind = 'bar', ax=ax)

    # Set plot properties
    ax.set_xlabel('Country')
    ax.set_title(indicator)

    plt.show()
    

#Bar graph 1
# Define the list of countries and years
countries = ['India', 'South Africa', 'United Kingdom', 'Pakistan',\
             'China', 'Japan', 'United States', 'Brazil', 'Germany',\
                 'Russian Federation', 'Australia']
years = ['1995', '1996', '1997', '1998', '1999', '2000', '2001', '2002', \
         '2003', '2004', '2005']
indicator = 'Population, total'


#Bar graph 2
# Call the function to plot the grouped bar graph
plot_grouped_bar_graph(df, countries, years, indicator)

# Define the list of countries and years
countries = ['India', 'South Africa', 'United Kingdom', 'Pakistan', 'China', \
             'Japan', 'United States', 'Brazil', 'Germany', \
                 'Russian Federation', 'Australia']
years = ['1995', '1996', '1997', '1998', '1999', '2000', '2001', '2002', \
         '2003', '2004', '2005']
indicator = 'CO2 emissions (kt)'

# Call the function to plot the grouped bar graph
plot_grouped_bar_graph(df, countries, years, indicator)





#import scipy.stats as stats
#for the statistical functions
df_filtered = df[df['Indicator Name'] == 'CO2 emissions from liquid fuel consumption (% of total)']
df_filtered = df_filtered[df_filtered['Country Name'].isin(countries)]
df_filtered = df_filtered.loc[:,'1995':'2005']

# calculate mean and standard deviation for each country using scipy
for country in countries:
    country_data = df_filtered.loc[df_filtered.index == country]
    if not country_data.empty:
        country_data_list = country_data.values.tolist()[0]
        mean = stats.mean(country_data_list)
        stdev = stats.stdev(country_data_list)
        print(country + ": Mean = " + str(mean) + ", Standard Deviation = " + str(stdev))


#Plotting line graph 1
plt.figure(figsize = (20,6))
plt.plot(df_filtered.transpose(),linestyle='--')
plt.legend(countries,bbox_to_anchor=(1.02, 1), loc='upper left',fontsize='small')
plt.xlabel('Year')
plt.title('CO2 emissions from liquid fuel consumption (% of total)')
plt.show()

#import scipy.stats as stats
#for the statistical functions
df_filtered = df[df['Indicator Name'] == 'CO2 emissions from gaseous fuel consumption (% of total)']
df_filtered = df_filtered[df_filtered['Country Name'].isin(countries)]
df_filtered = df_filtered.loc[:,'1995':'2005']


# calculate median and standard deviation for each country using scipy
for country in countries:
    country_data = df_filtered.loc[df_filtered.index == country]
    if not country_data.empty:
        country_data_list = country_data.values.tolist()[0]
        median = stats.median(country_data_list)
        stdev = stats.stdev(country_data_list)
        print(country + ": Median = " + str(median) + ", Standard Deviation = " + str(stdev))
        
# Set the figure size
plt.figure(figsize = (20,6))

# Plot the transposed dataframe with dashed lines
plt.plot(df_filtered.transpose(), linestyle='--')

# Add a legend with the country names
plt.legend(countries,bbox_to_anchor=(1.02, 1), loc='upper left',fontsize='small')

# label for the x-axis
plt.xlabel('Year')

# title for the plot
plt.title('CO2 emissions from gaseous fuel consumption (% of total)')



# Display the plot
plt.show()
