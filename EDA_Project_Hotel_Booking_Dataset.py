# Step 1: Importing Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# Step 2: Loading the Dataset
df = pd.read_csv('hotel_bookings 2.csv')

# Step 3: Overview of Data
print("Dataset Information:")
print(df.info())

# Step 4: Exploratory Data Analysis and Data Cleaning
print("\nFirst 5 Rows:")
print(df.head())

print("\nLast 5 Rows:")
print(df.tail())

# Checking column data types
print("\nColumns and Data Types:")
print(df.dtypes)

print(df.columns)

# Converting 'reservation_status_date' to datetime
df["reservation_status_date"] = pd.to_datetime(df["reservation_status_date"], format='%d/%m/%Y', errors='coerce')

# Summary of categorical variables
print("\nCategorical Column Summary:")
print(df.describe(include='object'))

# Checking unique values in categorical columns
for col in df.describe(include='object').columns:
    print(f"\nUnique values in {col}:")
    print(df[col].unique())
    print('-'*50)

# Checking missing values
print("\nMissing Values:")
print(df.isnull().sum())

# Dropping unnecessary columns and handling missing values
df.drop(['company', 'agent'], axis=1, inplace=True)
df.dropna(inplace=True)

# Verifying missing values are removed
print("\nMissing Values After Cleaning:")
print(df.isnull().sum())

# Statistical summary of numerical columns
print("\nStatistical Summary of Numerical Data:")
print(df.describe())

# Removing outliers in 'adr' column
df = df[df['adr'] < 5000]

# Step 5: Data Analysis
# Checking the proportion of cancellations
if 'is_canceled' in df.columns:
    cancelled_perc = df['is_canceled'].value_counts(normalize=True)
    print("\nCancellation Percentage:")
    print(cancelled_perc)
else:
    print("Column for cancellation status not found in the dataset.")

# Step 6: Filtering Data
# Resort Hotel
print("\nResort Hotel Data:")
resort_hotel = df[df['hotel'] == 'Resort Hotel']
print(resort_hotel['is_canceled'].value_counts(normalize=True))

# City Hotel
print("\nCity Hotel Data:")
city_hotel = df[df['hotel'] == 'City Hotel']
print(city_hotel['is_canceled'].value_counts(normalize=True))

# Step 7: Grouping Data
# Grouping ADR by reservation date
resort_hotel_adr = resort_hotel.groupby('reservation_status_date')[['adr']].mean()
city_hotel_adr = city_hotel.groupby('reservation_status_date')[['adr']].mean()

# Step 8: Market Segment Analysis
print("\nMarket Segment Distribution:")
print(df['market_segment'].value_counts())


cancelled_data = df[df['is_canceled'] == 1]

print("\nMarket Segment Cancellation Rate:")
print(cancelled_data['market_segment'].value_counts(normalize=True))


# Step 9: ADR Trends in Canceled vs. Not Canceled Bookings
cancelled_data = df[df['is_canceled'] == 1]
cancelled_df_adr = cancelled_data.groupby('reservation_status_date')[['adr']].mean().reset_index()
not_cancelled_data = df[df['is_canceled'] == 0]
not_cancelled_df_adr = not_cancelled_data.groupby('reservation_status_date')[['adr']].mean().reset_index()

# Step 10: Data Visualization
# Visualization: Reservation Status Count
plt.figure(figsize=(5,4))
plt.title('Reservation Status Count')
plt.bar(['Not Canceled', 'Canceled'], df['is_canceled'].value_counts(), edgecolor='k', width=0.7)
plt.show()

# Visualization: Reservation Status by Hotel Type
plt.figure(figsize=(8,4))
ax1 = sns.countplot(x='hotel', hue='is_canceled', data=df, palette='Blues')
ax1.legend(title='Cancellation Status', labels=['Not Canceled', 'Canceled'], bbox_to_anchor=(1,1))
plt.title('Reservation Status by Hotel Type')
plt.xlabel('Hotel Type')
plt.ylabel('Number of Reservations')
plt.show()

# Plotting ADR trends
plt.figure(figsize=(20,8))
plt.title('Average Daily Rate in City and Resort Hotels', fontsize=20)
plt.plot(resort_hotel_adr.index, resort_hotel_adr['adr'], label='Resort Hotel')
plt.plot(city_hotel_adr.index, city_hotel_adr['adr'], label='City Hotel')
plt.legend(fontsize=12)
plt.show()

df['year'] = df['reservation_status_date'].dt.year
# Reservation Status per Year
plt.figure(figsize=(16, 8))
ax1 = sns.countplot(x='year', hue='is_canceled', data=df, palette='bright')
ax1.legend(title='Cancellation Status', labels=['Not Canceled', 'Canceled'], bbox_to_anchor=(1, 1))
plt.title('Reservation Status per Year')
plt.xlabel('Year')
plt.ylabel('Number of Reservations')
plt.show()


# ADR per Year for Canceled Reservations
plt.figure(figsize=(15, 8))
plt.title('ADR per Year for Canceled Reservations', fontsize=20)
sns.barplot(x='year', y='adr', data=df[df['is_canceled'] == 1].groupby('year')[['adr']].sum().reset_index())
plt.show()


# Monthly Reservation Analysis
df['month'] = df['reservation_status_date'].dt.month
plt.figure(figsize=(16,8))
ax1 = sns.countplot(x='month', hue='is_canceled', data=df, palette='bright')
ax1.legend(title='Cancellation Status', labels=['Not Canceled', 'Canceled'], bbox_to_anchor=(1,1))
plt.title('Reservation Status per Month')
plt.xlabel('Month')
plt.ylabel('Number of Reservations')
plt.show()

# ADR per Month for Canceled Reservations
plt.figure(figsize=(15,8))
plt.title('ADR per Month for Canceled Reservations', fontsize=20)
sns.barplot(x='month', y='adr', data=df[df['is_canceled'] == 1].groupby('month')[['adr']].sum().reset_index())
plt.show()

# Top 10 Countries with Highest Cancellations
top_10_countries = cancelled_data['country'].value_counts()[:10]
plt.figure(figsize=(8,8))
plt.title('Top 10 Countries with Reservation Cancellations')
plt.pie(top_10_countries, autopct='%.2f', labels=top_10_countries.index)
plt.show()

# Average Daily Rate for Canceled vs. Not Canceled Reservations
plt.figure(figsize=(20,6))
plt.title('Average Daily Rate (ADR) for Canceled vs. Not Canceled Reservations')
plt.plot(not_cancelled_df_adr['reservation_status_date'], not_cancelled_df_adr['adr'], label='Not Canceled')
plt.plot(cancelled_df_adr['reservation_status_date'], cancelled_df_adr['adr'], label='Canceled')
plt.legend()
plt.show()

# Key Questions:
print("----------What is the cancellation rate?------------")
cancellation_rate = df['is_canceled'].mean() * 100
print(f"Overall Cancellation Rate: {cancellation_rate:.2f}%")

print("----------Which hotel type has more cancellations?-------------")
cancellation_by_hotel = df.groupby('hotel')['is_canceled'].mean() * 100
print(cancellation_by_hotel)

print("-----------Which month has the highest number of cancellations?---------")
df['reservation_status_date'] = pd.to_datetime(df['reservation_status_date'])
df['month'] = df['reservation_status_date'].dt.month
cancellations_by_month = df[df['is_canceled'] == 1]['month'].value_counts().sort_index()
print(cancellations_by_month)

print("----------Which country has the most cancellations?------------")
cancelled_countries = df[df['is_canceled'] == 1]['country'].value_counts().head(10)
print(cancelled_countries)

print("-----------How does ADR (Average Daily Rate) vary for canceled vs. non-canceled bookings?---------")
adr_comparison = df.groupby('is_canceled')['adr'].mean()
print(adr_comparison)

print("---------What is the most common market segment for bookings?----------")
market_segment_counts = df['market_segment'].value_counts()
print(market_segment_counts)

print("----------Does lead time impact cancellation rate?----------")
plt.figure(figsize=(10,5))
sns.histplot(data=df, x='lead_time', hue='is_canceled', bins=50, kde=True)
plt.title("Lead Time vs Cancellation Rate")
plt.show()
