import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load the GDP data
#-----------------------------------------------------------------------------------------------------------------------
gdp_data = pd.read_excel('C:\\Users\\teomeo\\Desktop\\aYEAR4\\semester 2\\Data Analysis\\DPII\\GDP\\GDP_per_Capita_clean.xlsx',header=0)
# Print the first few rows to check the column names
print("GDP Data")
print(gdp_data.head())

# Filter the GDP data for the year 2023
gdp_2019 = gdp_data[['Country', '2019']].copy()

# Check for missing values
print("\nMissing Values in GDP Data")
print(gdp_2019.isnull().sum())

# Check the extracted data
print("\nGDP 2019")
print(gdp_2019.head())

# Identifying the outliers in the GDP data using describe() method
print("\nGDP Data Description")
print(gdp_2019['2019'].describe())

# Plotting boxplots for GDP per Capita
plt.figure(figsize=(10, 5))
plt.boxplot(gdp_2019['2019'], vert=False)  
plt.title('Boxplot for GDP per Capita')
plt.xlabel('GDP per Capita (in Euros)')
plt.yticks([])
plt.show()

# Bar plot for GDP per Capita
gdp_2019_sorted = gdp_2019.sort_values('2019', ascending=False)

plt.figure(figsize=(15, 10))
bar_plot = sns.barplot(x='2019', y='Country', data=gdp_2019_sorted)

plt.title('GDP per Capita by Country in 2019')
plt.xlabel('GDP per Capita (in Euros)')
plt.ylabel('Country')

# set intervals based on the range of the data
max_gdp = gdp_2019_sorted['2019'].max()
tick_interval = 10000 # interval of 10k
ticks = list(range(0, int(max_gdp) + tick_interval, tick_interval))
bar_plot.set_xticks(ticks)
bar_plot.set_xticklabels([f"{int(x/1000)}k" for x in ticks])  # Labeling as 10k, 20k, etc.

# Adding the GDP values beside each bar
for index, value in enumerate(gdp_2019_sorted['2019']):
    plt.text(value, index, f'€{value:,.0f}', va='center')  # Format with commas, add Euro symbol
plt.show()
#-----------------------------------------------------------------------------------------------------------------------
#-----------------------------------------------------------------------------------------------------------------------

# Load the Employment Rate data
#-----------------------------------------------------------------------------------------------------------------------
employment_data = pd.read_excel('C:\\Users\\teomeo\\Desktop\\aYEAR4\\semester 2\\Data Analysis\\DPII\\EmploymentRates\\employment_rate_citizenship_clean.xlsx')
print("\nEmployment Data")
print(employment_data.head())
# Filter the Employment Rate data for the year 2023
employment_2019 = employment_data[['Country', '2019']].copy()

# Check for missing values
print("\nMissing Values in Employment Data")
print(employment_2019.isnull().sum())

# Check the extracted data
print("\nEmployment 2022")
print(employment_2019.head())

# Identifying the outliers in the Employment Rate data using describe() method
print("\nEmployment Data Description")
print(employment_2019['2019'].describe())

# Plotting boxplots for Employment Rate
plt.figure(figsize=(10, 5))
plt.boxplot(employment_2019['2019'], vert=False)
plt.title('Boxplot for Employment Rate')
plt.xlabel('Employment Rate (%)')
plt.yticks([]) 
plt.show()

# Bar plot for Employment Rate
# sorted by employment rate for 2019
employment_2019_sorted = employment_2019.sort_values('2019', ascending=False)
plt.figure(figsize=(15, 10))
bar_plot = sns.barplot(x='2019', y='Country', data=employment_2019_sorted)
plt.title('Employment Rate by Country in 2019')
plt.xlabel('Employment Rate (%)')
plt.ylabel('Country')

# Setting x-axis to display every 5% interval
max_rate = 100  #max employment rate can't be over 100%
tick_interval = 5
ticks = list(range(0, max_rate + tick_interval, tick_interval))
bar_plot.set_xticks(ticks)
bar_plot.set_xticklabels([f"{x}%" for x in ticks])  # Labeling as 5%, 10%, etc.

# Adding the employment rate values beside each bar
for index, value in enumerate(employment_2019_sorted['2019']):
    plt.text(value, index, f'{value:.1f}%', va='center')  # Format with one decimal place

plt.show()

#-----------------------------------------------------------------------------------------------------------------------
#-----------------------------------------------------------------------------------------------------------------------

#Load the Population data
#-----------------------------------------------------------------------------------------------------------------------
population_data = pd.read_excel('C:\\Users\\teomeo\\Desktop\\aYEAR4\\semester 2\\Data Analysis\\DPII\\Population\\population_clean.xlsx')
print("\nPopulation Data")
print(population_data.head())
# Filter the Population data for the year 2019
population_2019 = population_data[['Country', '2019']].copy()

# Check for missing values
print("\nMissing Values in Population Data")
print(population_2019.isnull().sum())

# Check the extracted data
print("\nPopulation 2019")
print(population_2019.head())

# Identifying the outliers in the Population data using describe() method
print("\nPopulation Data Description")
print(population_2019['2019'].describe())

# Plotting boxplots for Population
plt.figure(figsize=(10, 5))
plt.boxplot(population_2019['2019'], vert=False)
plt.title('Boxplot for Population')
plt.xlabel('Population(Millions)')
plt.yticks([])
plt.show()

# Bar plot for Population
# Sorting data for better visual arrangement
population_2019_sorted = population_2019.sort_values('2019', ascending=False)
plt.figure(figsize=(15, 10))
bar_plot = sns.barplot(x='2019', y='Country', data=population_2019_sorted)
plt.title('Population by Country in 2019')
plt.xlabel('Population (Millions)')
plt.ylabel('Country')

# Set x-axis to display every 10 million interval
bar_plot.set_xticks([i * 10_000_000 for i in range(9)])  # 0 to 80 million
bar_plot.set_xticklabels([f"{i*10}M" for i in range(9)])  # Labeling as 10M, 20M, etc.

# Adding the population values beside each bar
for index, value in enumerate(population_2019_sorted['2019']):
    plt.text(value, index, f'{value:,.0f}', va='center')  # Format with commas

plt.show()

#-----------------------------------------------------------------------------------------------------------------------
#-----------------------------------------------------------------------------------------------------------------------

# Load the Education level data
#-----------------------------------------------------------------------------------------------------------------------
# Load each dataset
level_3_4_data = pd.read_excel('C:\\Users\\teomeo\\Desktop\\aYEAR4\\semester 2\\Data Analysis\\DPII\\EducationLevel\\level_3_4.xlsx') # for education level 3-4 Upper secondary and post-secondary non-tertiary education (levels 3-4)
level_3_8_data = pd.read_excel('C:\\Users\\teomeo\\Desktop\\aYEAR4\\semester 2\\Data Analysis\\DPII\\EducationLevel\\level_3_8.xlsx') # for education level 3-8 Upper secondary, post-secondary non-tertiary and tertiary education (levels 3-8)

# Filter each dataset for the year of interest
column_of_interest = '2019'
level_3_4_data = level_3_4_data[['Country', column_of_interest]].copy()
level_3_8_data = level_3_8_data[['Country', column_of_interest]].copy()

#Check for missing values
print("\nMissing Values in Education Level 3-4 Data")
print(level_3_4_data.isnull().sum())
print("\nMissing Values in Education Level 3-8 Data")
print(level_3_8_data.isnull().sum())

print("\nEducation Level 3-4")
print(level_3_4_data.head())
print("\nEducation Level 3-8")
print(level_3_8_data.head())

# Identify the outliers in the Education Level data using describe() method
print("\nEducation Level 3-4 Description")
print(level_3_4_data[column_of_interest].describe())
print("\nEducation Level 3-8 Description")
print(level_3_8_data[column_of_interest].describe())

# Plotting boxplots for Education Level
plt.figure(figsize=(10, 5))
plt.boxplot(level_3_4_data[column_of_interest], vert=False)
plt.title('Boxplot for Education Level 3-4 - Upper secondary and post-secondary non-tertiary')
plt.xlabel('Education level (%)')
plt.yticks([])
plt.show()

plt.figure(figsize=(10, 5))
plt.boxplot(level_3_8_data[column_of_interest], vert=False)
plt.title('Boxplot for Education Level 3-8 - Upper secondary, post-secondary non-tertiary and tertiary')
plt.xlabel('Education level (%)')
plt.yticks([])
plt.show()

# Bar plots for Education Level Level 3-4 and Level 3-8
# For Education Level 3-4
plt.figure(figsize=(15, 10))
education_3_4_sorted = level_3_4_data.sort_values(column_of_interest, ascending=False)
bar_plot_3_4 = sns.barplot(x=column_of_interest, y='Country', data=education_3_4_sorted)

plt.title('Education Level 3-4 - Upper secondary and post-secondary non-tertiary by Country in 2019')
plt.xlabel('Education Level (%)')
plt.ylabel('Country')

# Setting x-axis to display every 10% interval 
max_rate_3_4 = 100  
ticks_3_4 = list(range(0, int(max_rate_3_4) + 10, 10))
bar_plot_3_4.set_xticks(ticks_3_4)
bar_plot_3_4.set_xticklabels([f"{x}%" for x in ticks_3_4])

# Adding the education level values beside each bar
for index, value in enumerate(education_3_4_sorted[column_of_interest]):
    plt.text(value, index, f'{value:.1f}%', va='center')

plt.show()

# For Education Level 3-8
plt.figure(figsize=(15, 10))
education_3_8_sorted = level_3_8_data.sort_values(column_of_interest, ascending=False)
bar_plot_3_8 = sns.barplot(x=column_of_interest, y='Country', data=education_3_8_sorted)
plt.title('Education Level 3-8 - Upper secondary, post-secondary non-tertiary and tertiary by Country in 2019')
plt.xlabel('Education Level (%)')
plt.ylabel('Country')

# Setting x-axis to display every 10% interval 
max_rate_3_8 = 100  
ticks_3_8 = list(range(0, int(max_rate_3_8) + 10, 10))
bar_plot_3_8.set_xticks(ticks_3_8)
bar_plot_3_8.set_xticklabels([f"{x}%" for x in ticks_3_8])

# Adding the education level values beside each bar
for index, value in enumerate(education_3_8_sorted[column_of_interest]):
    plt.text(value, index, f'{value:.1f}%', va='center')

plt.show()

# Rename the column to 'Percentage' for consistency
level_3_4_data.rename(columns={column_of_interest: 'Percentage'}, inplace=True)
level_3_8_data.rename(columns={column_of_interest: 'Percentage'}, inplace=True)

# Merge the datasets on the 'Country' column
education_data = pd.merge(level_3_4_data, level_3_8_data, on='Country', how='inner', suffixes=('_3_4', '_3_8'))

# Convert the 'Percentage' columns to numeric
education_data['Percentage_3_4'] = pd.to_numeric(education_data['Percentage_3_4'], errors='coerce')
education_data['Percentage_3_8'] = pd.to_numeric(education_data['Percentage_3_8'], errors='coerce')

print("\nEducation Data")
print(education_data.dtypes)

# Define weights for each education level
weights = {
    'Percentage_3_4': 0.5,
    'Percentage_3_8': 0.5
}

# Calculate the weighted average of the education levels
education_data['Education_Level'] = (education_data['Percentage_3_4'] * weights['Percentage_3_4'] +
                                education_data['Percentage_3_8'] * weights['Percentage_3_8'])

# Normalize the education index to a scale of 0 to 1
education_data['Normalized_Education_Level'] = (
    (education_data['Education_Level'] - education_data['Education_Level'].min()) / 
    (education_data['Education_Level'].max() - education_data['Education_Level'].min())
)
# Check the merged dataset
print("\nEducation Data")
print(education_data)
 
# Sort the data by 'Education_Level' in descending order for better visualization
education_data_sorted = education_data.sort_values('Education_Level', ascending=False)

plt.figure(figsize=(15, 10))
bar_plot = sns.barplot(x='Education_Level', y='Country', data=education_data_sorted)
plt.title('Education Level Composite Index by Country in 2019')
plt.xlabel('Education Level (%)')
plt.ylabel('Country')

# Adding the education level values beside each bar
for index, (value, country) in enumerate(zip(education_data_sorted['Education_Level'], education_data_sorted['Country'])):
    plt.text(value, index, f'{value:.1f}%', va='center')  # Format with one decimal place and percentage sign

plt.show()

# Loanding the Foreign population data
#-----------------------------------------------------------------------------------------------------------------------
foreign_population_data = pd.read_excel('C:\\Users\\teomeo\\Desktop\\aYEAR4\\semester 2\\Data Analysis\\DPII\\ForeignPopulation\\foreign_population.xlsx')
print("Foreign Population Data")
print(foreign_population_data.head())

# Filter the Foreign Population data for the year 2019
foreign_population = foreign_population_data[['Country', 'Foreigners%']].copy()

# Check for missing values
print("\nMissing Values in Foreign Population Data")
print(foreign_population.isnull().sum())

# Check the extracted data
print("\nForeign Population 2019")
print(foreign_population.head())

# Identifying the outliers in the Foreign Population data using describe() method
print("\nForeign Population Data Description")
print(foreign_population['Foreigners%'].describe())

# Plotting boxplots for Foreign Population
plt.figure(figsize=(10, 5))
plt.boxplot(foreign_population['Foreigners%'], vert=False)
plt.title('Boxplot for Foreign Population')
plt.xlabel('Foreign Population (%)')
plt.yticks([])
plt.show()

# Bar plot for Foreign Population
# Sort the data by 'Foreigners%' in descending order for better visualization
foreign_population_sorted = foreign_population.sort_values('Foreigners%', ascending=False)

plt.figure(figsize=(15, 10))
bar_plot = sns.barplot(x='Foreigners%', y='Country', data=foreign_population_sorted)
plt.title('Foreign Population by Country in 2019')
plt.xlabel('Foreign Population (%)')
plt.ylabel('Country')

# Adding the foreign population values beside each bar
for index, (value, country) in enumerate(zip(foreign_population_sorted['Foreigners%'], foreign_population_sorted['Country'])):
    plt.text(value, index, f'{value:.1f}%', va='center')  # Format with one decimal place and percentage sign

plt.show()

# Normalize the foreign population index to a scale of 0 to 1
foreign_population['Normalized_Foreign_Population'] = (foreign_population['Foreigners%'] - foreign_population['Foreigners%'].min()) / (foreign_population['Foreigners%'].max() - foreign_population['Foreigners%'].min())

# Loading Health Care data
#-----------------------------------------------------------------------------------------------------------------------
health_care_data = pd.read_excel('C:\\Users\\teomeo\\Desktop\\aYEAR4\\semester 2\\Data Analysis\\DPII\\HealthCareAccess\\healthcare_expenditure_clean.xlsx')
print("\nHealth Care Data")
print(health_care_data.head())

# Filter the Health Care data for the year 2019
health_care_2019 = health_care_data[['Country', '2019']].copy()
health_care_2019.rename(columns={'2019': 'Healthcare_Expenditure'}, inplace=True)

# Check for missing values
print("\nMissing Values in Health Care Data")
print(health_care_2019.isnull().sum())

# Check the extracted data
print("\nHealth Care 2019")
print(health_care_2019.head())

# Identifying the outliers in the Health Care data using describe() method
print("\nHealth Care Data Description")
print(health_care_2019['Healthcare_Expenditure'].describe())

# Plotting boxplots for Health Care
plt.figure(figsize=(10, 5))
plt.boxplot(health_care_2019['Healthcare_Expenditure'], vert=False)
plt.title('Boxplot for Health Care Expenditure per Inhabitant')
plt.xlabel('Health Care Expenditure (in Euros)')
plt.yticks([])
plt.show()

# Bar plot for Health Care Expenditure
# Sort the data by 'Healthcare_Expenditure' in descending order for better visualization
health_care_sorted = health_care_2019.sort_values('Healthcare_Expenditure', ascending=False)

plt.figure(figsize=(15, 10))
bar_plot = sns.barplot(x='Healthcare_Expenditure', y='Country', data=health_care_sorted)
plt.title('Health Care Expenditure by Country per Inhabitant in 2019')
plt.xlabel('Health Care Expenditure (in Euros)')
plt.ylabel('Country')

# Adding the health care expenditure values beside each bar
for index, (value, country) in enumerate(zip(health_care_sorted['Healthcare_Expenditure'], health_care_sorted['Country'])):
    plt.text(value, index, f'€{value:,.0f}', va='center')  # Format with commas for thousands and Euro symbol

plt.show()

# Normalize the Health Care index to a scale of 0 to 1
health_care_2019['Normalized_Health_Care'] = (health_care_2019['Healthcare_Expenditure']
                                              - health_care_2019['Healthcare_Expenditure'].min()) / (health_care_2019['Healthcare_Expenditure'].max()
                                            - health_care_2019['Healthcare_Expenditure'].min())

#Loading the Innovation data
#-----------------------------------------------------------------------------------------------------------------------
innovation_data = pd.read_excel('C:\\Users\\teomeo\\Desktop\\aYEAR4\\semester 2\\Data Analysis\\DPII\\InnovationIndex\\innovation_index.xlsx')
print("\nInnovation Data")
print(innovation_data.head())

# Filter the Innovation data for the year 2019
innovation_index = innovation_data[['Country', 'InnovationIndex']].copy()

# Check for missing values
print("\nMissing Values in Innovation Data")
print(innovation_index.isnull().sum())

# Check the extracted data
print("\nInnovation Index 2019")
print(innovation_index.head())

# Identifying the outliers in the Innovation data using describe() method
print("\nInnovation Data Description")
print(innovation_index['InnovationIndex'].describe())

# Plotting boxplots for Innovation Index
plt.figure(figsize=(10, 5))
plt.boxplot(innovation_index['InnovationIndex'], vert=False)
plt.title('Boxplot for Innovation Index')
plt.xlabel('Innovation Index (0-100 Points)')
plt.yticks([])
plt.show()

# Bar plot for Innovation Index
# Sort the data by 'InnovationIndex' in descending order for better visualization
innovation_index_sorted = innovation_index.sort_values('InnovationIndex', ascending=False)

plt.figure(figsize=(15, 10))

bar_plot = sns.barplot(x='InnovationIndex', y='Country', data=innovation_index_sorted)
plt.title('Innovation Index by Country in 2019')
plt.xlabel('Innovation Index (0-100 Points)')
plt.ylabel('Country')

# Adding the innovation index values beside each bar
for index, (value, country) in enumerate(zip(innovation_index_sorted['InnovationIndex'], innovation_index_sorted['Country'])):
    plt.text(value, index, f'{value:.1f} points', va='center')  # Format with one decimal place and add 'points' for clarity

plt.show()

# Convert the 'InnovationIndex' column to numeric
innovation_index['InnovationIndex'] = pd.to_numeric(innovation_index['InnovationIndex'], errors='coerce')

# Normalize the Innovation index to a scale of 0 to 1
innovation_index['Normalized_Innovation_Index'] = (innovation_index['InnovationIndex'] - innovation_index['InnovationIndex'].min()) / (innovation_index['InnovationIndex'].max() - innovation_index['InnovationIndex'].min())

 
#-----------------------------------------------------------------------------------------------------------------------
# Merge the GDP, Employment, Population data, Foreign Population data, Education data and Health care data
merged_data = pd.merge(gdp_2019, employment_2019, on='Country', how='inner', suffixes=('_GDP', '_Employment'))
merged_data = pd.merge(merged_data, population_2019, on='Country', how='inner')
merged_data.columns = ['Country', 'GDP', 'Employment', 'Population']
# Merge the Foreign Population data
merged_data = pd.merge(merged_data, foreign_population[['Country', 'Foreigners%']], on='Country', how='inner')
# Merge the Education data
merged_data = pd.merge(merged_data, education_data[['Country', 'Education_Level']], on='Country', how='inner')
# Merge the Health Care data
merged_data = pd.merge(merged_data, health_care_2019[['Country','Healthcare_Expenditure']], on='Country', how='inner')

# Merge the Innovation data
merged_data = pd.merge(merged_data, innovation_index[['Country', 'InnovationIndex']], on='Country', how='inner')

# Set display options to show all rows and columns
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)

# Check the merged DataFrame
print("\nMerged Data")
print(merged_data)


# Normalizing GDP, Employment, Population
# merged_data['Normalized_GDP'] = (merged_data['GDP'] - merged_data['GDP'].min()) / (merged_data['GDP'].max() - merged_data['GDP'].min())
# merged_data['Normalized_Employment'] = (merged_data['Employment'] - merged_data['Employment'].min()) / (merged_data['Employment'].max() - merged_data['Employment'].min())
# merged_data['Normalized_Population'] = (merged_data['Population'] - merged_data['Population'].min()) / (merged_data['Population'].max() - merged_data['Population'].min())

# # merge Education, GDP, Employment, Population
# merged_data['DPII_Index'] = (merged_data['Normalized_Education_Level'] + merged_data['Normalized_GDP'] + merged_data['Normalized_Employment'] + merged_data['Normalized_Population']) / 4

# # Set display options to show all rows and columns
# pd.set_option('display.max_rows', None)
# pd.set_option('display.max_columns', None)
# pd.set_option('display.width', 1000)

# # Display the final merged data with the composite index
# print("\nFinal Merged Data with Composite Index")
# print(merged_data[['Country', 'Normalized_Education_Level', 'Normalized_GDP', 'Normalized_Employment', 'Normalized_Population', 'DPII_Index']])

# # Quick visualization of the Composite Index
# sns.histplot(merged_data['DPII_Index'], kde=True)
# plt.title('Distribution of DPII Index')
# plt.xlabel('Composite Index Score')
# plt.ylabel('Frequency')
# plt.show()


# Optional: Check for countries not present in all datasets
# countries_gdp = set(gdp_2019['Country'])
# countries_employment = set(employment_2019['Country'])
# countries_population = set(population_2019['Country'])

# print("Countries in GDP but not in Employment or Population:", countries_gdp - countries_employment - countries_population)
# print("Countries in Employment but not in GDP or Population:", countries_employment - countries_gdp - countries_population)
# print("Countries in Population but not in GDP or Employment:", countries_population - countries_gdp - countries_employment)



# pd.set_option('display.max_rows', None)
# # Set the option to display all columns if necessary
# pd.set_option('display.max_columns', None)

# print(merged_data)
# Calculate the DPII index
