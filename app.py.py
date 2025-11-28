import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
plt.style.use('dark_background')

df = pd.read_csv('zomato.csv')
df.head()

df.shape

df.columns

df.info()

# 2. Clean cuisines column
# ----------------------------------------
df['cuisines'] = df['cuisines'].astype(str).str.lower().str.strip()
df['cuisines'] = df['cuisines'].fillna("unknown")

# Replace separators with space
df['cuisines'] = df['cuisines'].str.replace(',', ' ')
df['cuisines'] = df['cuisines'].str.replace('/', ' ')

# Split cuisines into list
df['cuisine_list'] = df['cuisines'].str.split()

# ----------------------------------------
# 3. Create dummy columns using explode + get_dummies (fast)
# ----------------------------------------
dummy = (
    df['cuisine_list']
    .explode()
    .str.strip()
    .str.get_dummies()
)

# Group back to original rows (max ensures Yes if any repeat)
dummy = dummy.groupby(level=0).max()

# Convert 1/0 into Yes/No
dummy = dummy.replace({1: "Yes", 0: "No"})

# ----------------------------------------
# 4. Add all cuisine columns at once using concat (no warnings)
# ----------------------------------------
df = pd.concat([df, dummy], axis=1)

# Remove temporary column
df = df.drop(columns=['cuisine_list'])

# ----------------------------------------
# 5. Show final output
# ----------------------------------------
print("\nAfter Preprocessing:")
display(df.head())
print("Final shape:", df.shape)

df.drop_duplicates(inplace = True)
df.shape

df['rate'].unique()

def handlerate(value):
    if(value=='NEW' or value=='-'):
        return np.nan
    else:
        value = str(value).split('/')
        value = value[0]
        return float(value)

df['rate'] = df['rate'].apply(handlerate)
df['rate'].head()

df['rate'].fillna(df['rate'].mean(), inplace = True)
df['rate'].isnull().sum()

df.info()

df.dropna(inplace = True)
df.head()

df.rename(columns = {'approx_cost(for two people)':'Cost2plates', 'listed_in(type)':'Type'}, inplace = True)
df.head()

df['location'].unique()

df['listed_in(city)'].unique()

df = df.drop(['listed_in(city)'], axis = 1)

df['Cost2plates'].unique()

def handlecomma(value):
    value = str(value)
    if ',' in value:
        value = value.replace(',', '')
        return float(value)
    else:
        return float(value)

df['Cost2plates'] = df['Cost2plates'].apply(handlecomma)
df['Cost2plates'].unique()

df.head()

rest_types = df['rest_type'].value_counts(ascending  = False)
rest_types

rest_types_lessthan1000 = rest_types[rest_types<1000]
rest_types_lessthan1000

def handle_rest_type(value):
    if(value in rest_types_lessthan1000):
        return 'others'
    else:
        return value

df['rest_type'] = df['rest_type'].apply(handle_rest_type)
df['rest_type'].value_counts()

location = df['location'].value_counts(ascending  = False)

location_lessthan300 = location[location<300]



def handle_location(value):
    if(value in location_lessthan300):
        return 'others'
    else:
        return value

df['location'] = df['location'].apply(handle_location)
df['location'].value_counts()

# Number of restaurants by location
plt.figure(figsize=(12,6))
top_locations = df['location'].value_counts().head(10)
sns.barplot(
    x=bottom_locations.values,
    y=bottom_locations.index,
    hue=bottom_locations.index,
    palette='mako',
    legend=False
)
plt.title('Top 10 Locations with Most Restaurants in Bengaluru')
plt.xlabel('Number of Restaurants')
plt.ylabel('Location')
plt.show()

# Underserved areas (fewest restaurants)
plt.figure(figsize=(12,6))
bottom_locations = df['location'].value_counts().tail(10)
sns.barplot(
    x=bottom_locations.values,
    y=bottom_locations.index,
    hue=bottom_locations.index,
    palette='mako',
    legend=False
)
plt.title('Bottom 10 Locations with Fewest Restaurants')
plt.xlabel('Number of Restaurants')
plt.ylabel('Location')
plt.show()

# Count most popular cuisines
plt.figure(figsize=(12,6))
top_cuisines = df['cuisines'].value_counts().head(10)
sns.barplot(x=top_cuisines.values, y=top_cuisines.index, palette='viridis')
plt.title('Top 10 Most Popular Cuisines in Bengaluru')
plt.xlabel('Number of Restaurants')
plt.ylabel('Cuisine Type')
plt.show()

# Cuisine diversity per area
cuisine_diversity = df.groupby('location')['cuisines'].nunique().sort_values(ascending=False)
plt.figure(figsize=(12,6))
sns.barplot(x=cuisine_diversity.head(10).values, y=cuisine_diversity.head(10).index)
plt.title('Top Areas with Highest Cuisine Diversity')
plt.xlabel('Unique Cuisine Types')
plt.ylabel('Location')
plt.show()

# ------------------------------
df['Cost2plates'] = (
    df['Cost2plates']
    .astype(str)
    .str.replace(',', '')
    .str.extract(r'(\d+\.?\d*)')[0]   # regex fixed with raw string
    .astype(float)
)

# ------------------------------
# 2. Clean 'rate' column
# ------------------------------
df['rate'] = (
    df['rate']
    .astype(str)
    .str.extract(r'(\d+\.?\d*)')[0]
    .astype(float)
)

# ------------------------------
# 3. Drop missing values
# ------------------------------
df.dropna(subset=['Cost2plates', 'rate'], inplace=True)

# ------------------------------
# 4. Average Cost for Two — Top 10 Locations
# ------------------------------
avg_cost = (
    df.groupby('location')['Cost2plates']
    .mean()
    .sort_values(ascending=False)
    .head(10)
)

plt.figure(figsize=(12,6))
sns.barplot(
    x=avg_cost.values,
    y=avg_cost.index,
    hue=avg_cost.index,
    legend=False
)
plt.title('Average Cost for Two by Location')
plt.xlabel('Avg Cost for Two')
plt.ylabel('Location')
plt.show()

# ------------------------------
# 5. Scatter Plot — Cost vs Rating
# ------------------------------
plt.figure(figsize=(8,5))
sns.scatterplot(
    x='Cost2plates',
    y='rate',
    data=df,
    alpha=0.5
)
plt.title('Cost vs Rating')
plt.xlabel('Cost for Two')
plt.ylabel('Rating')
plt.show()

# Highest and lowest rated restaurants
top_rated = df.sort_values('rate', ascending=False).head(10)
low_rated = df.sort_values('rate', ascending=True).head(10)

print("⭐ Top 10 Highly Rated Restaurants")
print(top_rated[['name','rate','location','cuisines']])

print("\n👎 Lowest Rated Restaurants")
print(low_rated[['name','rate','location','cuisines']])

# Rating vs Price correlation
plt.figure(figsize=(6,4))
sns.heatmap(df[['rate','Cost2plates']].corr(), annot=True, cmap='coolwarm')
plt.title('Correlation between Rating and Price')
plt.show()

# Restaurant type distribution
plt.figure(figsize=(10,6))
types = df['Type'].value_counts().head(10)
sns.barplot(x=types.values, y=types.index, palette='crest')
plt.title('Most Common Restaurant Types')
plt.xlabel('Count')
plt.ylabel('Type')
plt.show()

# Popular types by location
top_types = df.groupby('location')['Type'].agg(lambda x: x.mode()[0])
top_types.head(10)

