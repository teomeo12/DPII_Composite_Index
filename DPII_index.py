import pandas as pd

# Load the GDP data
gdp_data = pd.read_excel('C:\\Users\\teomeo\\Desktop\\aYEAR4\\semester 2\\Data Analysis\\DPII\\GDP\\GDP_per_Capita_clean.xlsx',header=0)
# Print the first few rows to check the column names
print("GDP Data")
print(gdp_data.head())

# Filter the GDP data for the year 2023
gdp_2019 = gdp_data[['Country', '2019']].copy()

# Check the extracted data
print("GDP 2019")
print(gdp_2019.head())

# Load the Employment Rate data
employment_data = pd.read_excel('C:\\Users\\teomeo\\Desktop\\aYEAR4\\semester 2\\Data Analysis\\DPII\\EmploymentRates\\employment_rate_citizenship_clean.xlsx')
print("Employment Data")
print(employment_data.head())
# Filter the Employment Rate data for the year 2023
employment_2019 = employment_data[['Country', '2019']].copy()
# Check the extracted data
print("Employment 2022")
print(employment_2019.head())

# Merge the GDP and Employment Rate data ,include countries that are present in both datasets.
merged_data = pd.merge(gdp_2019, employment_2019, on='Country', how='inner', suffixes=('_GDP', '_Employment'))
# Check the merged DataFrame
print("Merged Data")
print(merged_data.head())

# Countries in GDP but not in Employment
countries_not_in_employment = set(gdp_2019['Country']) - set(employment_2019['Country'])

# Countries in Employment but not in GDP
countries_not_in_gdp = set(employment_2019['Country']) - set(gdp_2019['Country'])

print("Countries in GDP but not in Employment:", countries_not_in_employment)
print("Countries in Employment but not in GDP:", countries_not_in_gdp)

pd.set_option('display.max_rows', None)
# Set the option to display all columns if necessary
pd.set_option('display.max_columns', None)

print(merged_data)
# Calculate the DPII index
