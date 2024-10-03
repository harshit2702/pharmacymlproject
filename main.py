import pandas as pd

# Load the uploaded CSV file to inspect its contents
file_path = '/workspaces/codespaces-jupyter/salesdaily.csv/train-salesdaily.csv.csv'
data = pd.read_csv(file_path)

# Display basic information about the dataset and the first few rows to understand its structure
data_info = data.info()
data_head = data.head()

print(data_info)
print(data_head)


import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose

# Ensure plots display correctly
import matplotlib
matplotlib.use('Agg')

# Convert 'datum' to datetime and set it as the index
data['datum'] = pd.to_datetime(data['datum'], format='%m/%d/%Y')
data.set_index('datum', inplace=True)

# Handle any missing dates by resampling (if necessary)
data = data.asfreq('D').fillna(method='ffill')

# Plot the time series for 'N02BE'
plt.figure(figsize=(12,6))
data['N02BE'].plot()
plt.title('Daily Sales of N02BE')
plt.xlabel('Date')
plt.ylabel('Sales')
plt.tight_layout()
plt.savefig('N02BE_timeseries.png')
plt.close()

from statsmodels.tsa.seasonal import seasonal_decompose

# Perform seasonal decomposition
decomposition = seasonal_decompose(data['N02BE'], model='additive', period=365)

# Extract trend, seasonal, and residual components
trend = decomposition.trend
seasonal = decomposition.seasonal
residual = decomposition.resid

# Save the decomposed components plots
decomposition.plot()
plt.tight_layout()
plt.savefig('N02BE_decomposition.png')
plt.close()

# Summarize the components
trend_summary = trend.describe()
seasonal_summary = seasonal.describe()
residual_summary = residual.describe()

# Output summaries
print(trend_summary, seasonal_summary, residual_summary)

import seaborn as sns


# Re-defining the sales_columns as it may not have persisted
sales_columns = ['M01AB', 'M01AE', 'N02BA', 'N02BE', 'N05B', 'N05C', 'R03', 'R06']

# Group data by 'Weekday Name' and calculate the mean sales for each weekday
weekday_sales = data.groupby('Weekday Name')[sales_columns].mean()

# Reorder the days for proper visualization (starting from Monday)
ordered_days = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
weekday_sales = weekday_sales.reindex(ordered_days)

# Plotting the average sales per weekday for each product category
plt.figure(figsize=(12,8))
sns.lineplot(data=weekday_sales, markers=True, dashes=False)
plt.title('Average Sales per Weekday for Each Product Category')
plt.xlabel('Day of the Week')
plt.ylabel('Average Sales')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('weekday_sales_fixed.png')
plt.close()
