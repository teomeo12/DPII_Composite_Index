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

# Check the extracted data
print("GDP 2019")
print(gdp_2019.head())
#-----------------------------------------------------------------------------------------------------------------------
#-----------------------------------------------------------------------------------------------------------------------

# Load the Employment Rate data
#-----------------------------------------------------------------------------------------------------------------------
employment_data = pd.read_excel('C:\\Users\\teomeo\\Desktop\\aYEAR4\\semester 2\\Data Analysis\\DPII\\EmploymentRates\\employment_rate_citizenship_clean.xlsx')
print("Employment Data")
print(employment_data.head())
# Filter the Employment Rate data for the year 2023
employment_2019 = employment_data[['Country', '2019']].copy()
# Check the extracted data
print("Employment 2022")
print(employment_2019.head())
#-----------------------------------------------------------------------------------------------------------------------
#-----------------------------------------------------------------------------------------------------------------------

#Load the Population data
#-----------------------------------------------------------------------------------------------------------------------
population_data = pd.read_excel('C:\\Users\\teomeo\\Desktop\\aYEAR4\\semester 2\\Data Analysis\\DPII\\Population\\population_clean.xlsx')
print("Population Data")
print(population_data.head())
# Filter the Population data for the year 2019
population_2019 = population_data[['Country', '2019']].copy()

# Check the extracted data
print("\nPopulation 2019")
print(population_2019.head())
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

print("\nEducation Level 3-4")
print(level_3_4_data.head())

# Rename the column to 'Percentage' for consistency
level_3_4_data.rename(columns={column_of_interest: 'Percentage'}, inplace=True)
level_3_8_data.rename(columns={column_of_interest: 'Percentage'}, inplace=True)

# Merge the datasets on the 'Country' column
education_data = pd.merge(level_3_4_data, level_3_8_data, on='Country', how='inner', suffixes=('_3_4', '_3_8'))

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

# Loanding the Foreign population data
#-----------------------------------------------------------------------------------------------------------------------
foreign_population_data = pd.read_excel('C:\\Users\\teomeo\\Desktop\\aYEAR4\\semester 2\\Data Analysis\\DPII\\ForeignPopulation\\foreign_population.xlsx')
print("Foreign Population Data")
print(foreign_population_data.head())

# Filter the Foreign Population data for the year 2019
foreign_population = foreign_population_data[['Country', 'Foreigners%']].copy()

# Normalize the foreign population index to a scale of 0 to 1
foreign_population['Normalized_Foreign_Population'] = (foreign_population['Foreigners%'] - foreign_population['Foreigners%'].min()) / (foreign_population['Foreigners%'].max() - foreign_population['Foreigners%'].min())


# Check the extracted data
print("\nForeign Population 2019")
print(foreign_population.head())
 
#-----------------------------------------------------------------------------------------------------------------------
# Merge the GDP, Employment, and Population data
merged_data = pd.merge(gdp_2019, employment_2019, on='Country', how='inner', suffixes=('_GDP', '_Employment'))
merged_data = pd.merge(merged_data, population_2019, on='Country', how='inner')
merged_data.columns = ['Country', 'GDP', 'Employment', 'Population']
# Merge the Foreign Population data
merged_data = pd.merge(merged_data, foreign_population[['Country', 'Foreigners%']], on='Country', how='inner')
# Merge the Education data
merged_data = pd.merge(merged_data, education_data[['Country', 'Normalized_Education_Level']], on='Country', how='inner')

# Set display options to show all rows and columns
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)

# Check the merged DataFrame
print("\nMerged Data")
print(merged_data)


# Normalizing GDP, Employment, Population
merged_data['Normalized_GDP'] = (merged_data['GDP'] - merged_data['GDP'].min()) / (merged_data['GDP'].max() - merged_data['GDP'].min())
merged_data['Normalized_Employment'] = (merged_data['Employment'] - merged_data['Employment'].min()) / (merged_data['Employment'].max() - merged_data['Employment'].min())
merged_data['Normalized_Population'] = (merged_data['Population'] - merged_data['Population'].min()) / (merged_data['Population'].max() - merged_data['Population'].min())

# merge Education, GDP, Employment, Population
merged_data['DPII_Index'] = (merged_data['Normalized_Education_Level'] + merged_data['Normalized_GDP'] + merged_data['Normalized_Employment'] + merged_data['Normalized_Population']) / 4

# Set display options to show all rows and columns
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)

# Display the final merged data with the composite index
print("\nFinal Merged Data with Composite Index")
print(merged_data[['Country', 'Normalized_Education_Level', 'Normalized_GDP', 'Normalized_Employment', 'Normalized_Population', 'DPII_Index']])

# Quick visualization of the Composite Index
sns.histplot(merged_data['DPII_Index'], kde=True)
plt.title('Distribution of DPII Index')
plt.xlabel('Composite Index Score')
plt.ylabel('Frequency')
plt.show()


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
