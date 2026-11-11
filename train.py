#!/usr/bin/env python
# coding: utf-8
import numpy as np
import pandas as pd
import json
import xgboost as xgb

from sklearn.metrics import root_mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction import DictVectorizer

ETA = 0.05
MAX_DEPTH = 6
MIN_CHILD_WEIGHT = 5
NUM_BOOST_ROUND = 301

df = pd.read_csv('data/listings.csv.gz')

# **Cleaning data**
to_drop_replaced = []
to_drop_low_info = []
to_drop_id = []
to_drop_leakage = []

# Delete data with empty or almost empty columns  
missing = df.isnull().sum()
columns_empty = missing[missing > len(df)*0.9].index
df = df.drop(columns=columns_empty)

# Delete data with missing price
df = df.drop(df[df.price.isnull()].index)
missing = df.isnull().sum()

# Delete the 3 rows with missing values in the same columns
cols = missing[missing == 3].index
df = df.dropna(subset=cols)

# Dropping the row with missing bathroom description
df = df.drop(df[df.bathrooms_text.isnull()].index)

# Cleaning price
df['price'] = df['price'].str.replace('$', '')
df['price'] = df['price'].str.replace(',', '')
df.price = pd.to_numeric(df.price)

# **Feature engineering**

# 1. Host location (categories)
def host_location_category(loc): 
    if pd.isna(loc): 
        return 'unknown' 
    if 'Thessaloniki' in loc: 
        return 'thessaloniki' 
    if'Greece' in loc: 
        return 'greece'
    return 'abroad' 


df['host_location_cat'] = df['host_location'].apply(host_location_category) 
to_drop_replaced.append('host_location')

# 2. Reviews
#    a) make a new column 'has_reviews'
#    b) 'reviews_per_month' fill with 0
def has_reviews_category(loc): 
    if loc == 0: 
        return 0 
    return 1 


df['has_reviews'] = df['number_of_reviews'].apply(has_reviews_category) 
df['reviews_per_month'] = df['reviews_per_month'].fillna(0)
# After splitting, fill all reviews columns with the median

# 3. License
def license_ok(loc): 
    if pd.isna(loc): 
        return 0
    return 1 


df['license_ok'] = df['license'].apply(license_ok) 

# 4. Host description (make a flag if host wrote the description)
df['host_has_descr'] = (df['host_about'].str.len().fillna(0) >= 20).astype(int)
to_drop_replaced.append('host_about')
to_drop_replaced.append('license')

# 5. Amenities. (Create a new feature number of amenities)
df['n_amenities'] = df['amenities'].apply(json.loads).str.len()
to_drop_replaced.append('amenities')

# Delete non unique columns
nunique = df.nunique()
columns_nonunique = nunique[nunique == 1].index
df = df.drop(columns=columns_nonunique)

# 6. First and last reviews. Cleaning dates (new columns - to count how many days from the first review and from the last, 
# the days will be counted from the last scraped date)
df['days_since_last_review'] = (pd.to_datetime(df.last_scraped) - pd.to_datetime(df.last_review)).dt.days
df['days_since_first_review'] = (pd.to_datetime(df.last_scraped) - pd.to_datetime(df.first_review)).dt.days
to_drop_replaced.append('last_scraped')
to_drop_replaced.append('last_review')
to_drop_replaced.append('first_review')

# 7. Working with bathrooms categories. 87% of listings have exactly 1 bath, so creating only  2 new flags - Bathroom numbers (1 if more than 1, 0 if 1 or less) and Shared bathroom (=1 if shared)
df['bathrooms_num'] = df['bathrooms_text'].str.extract(r'(\d+\.?\d*)')[0].astype(float)
df['bathrooms_num'] = df['bathrooms_num'].fillna(0.5)
df['bathrooms_num'] = (df['bathrooms_num'] > 1).astype(int)
df['bathrooms_shared'] = df['bathrooms_text'].str.contains('shared', case=False).astype(int)
to_drop_replaced.append('bathrooms_text')

# **Filtering rows and dropping columns**

# Delete prices > 1000. 2 prices about 10000 euros are clearly placeholders; values above 1000 euros look unrealistic for Thessaloniki
df = df.drop(df[df.price > 1000].index)

# Target: log transformation of price
df['log_price'] = np.log1p(df['price'])

# Id columns
id_cols = [c for c in df.columns if c == 'id' or c.endswith('_id')]

# Delete 17 listings because they are long-term rentals
df = df[df['minimum_nights'] <= 30]

# The column calculated_host_listings_count_shared_rooms to drop because only 8 non-zero values
to_drop_low_info.append('calculated_host_listings_count_shared_rooms')

# Adding to the Drop list all columns with 'nights' except 'minimum_nights' because others have around the same skew and outliers
night_cols = [c for c in df.columns if 'nights' in c and c != 'minimum_nights']
to_drop_low_info.extend(night_cols)

# We will aggregate 4 columns in 2, counting in month time as user and host and drop 4 old columns
df['host_tenure_month'] = df['hosts_time_as_host_months'] + df['hosts_time_as_host_years'] * 12
df['user_tenure_month'] = df['hosts_time_as_user_months'] + df['hosts_time_as_user_years'] * 12
host_user_columns = [c for c in df.columns if 'hosts_time_as_' in c]
to_drop_replaced.extend(host_user_columns)

# 'estimated_occupancy_l365d' derived from review counts with Inside Airbnb’s own assumptions; duplicates existing review features 
to_drop_low_info.append('estimated_occupancy_l365d')

# Drop all columns we don't need
to_drop_id = id_cols
to_drop_leakage.append('estimated_revenue_l365d')
df =  df.drop(columns=(to_drop_id + to_drop_leakage + to_drop_replaced + to_drop_low_info))

# Removing features that are highly correlated with each other
availability_cols = [c for c in df.columns if 'availability_' in c and c != 'availability_90']
reviews_cols = [c for c in df.columns if ('review_scores_' in c or 'number_of_reviews_l' in c) 
                and c != 'review_scores_rating' and c != 'review_scores_location']
to_drop_correlated = ['calculated_host_listings_count_entire_homes']
to_drop_correlated.extend(availability_cols)
to_drop_correlated.extend(reviews_cols)

df =  df.drop(columns=to_drop_correlated)

# Drop all price_quote columns as well
to_drop_price_quote = [c for c in df.columns if c.startswith('price_quote')]
df =  df.drop(columns=to_drop_price_quote)
to_drop_categorical_cols = [c for c in df.columns if c.endswith('_url')]
to_drop_categorical_cols.append('name')
to_drop_categorical_cols.append('host_name')
to_drop_categorical_cols.append('description')
to_drop_categorical_cols.append('calendar_last_scraped')
df = df.drop(columns=to_drop_categorical_cols)
to_drop_categorical_analysis = ['property_type'] 
df = df.drop(columns=to_drop_categorical_analysis) 

# **Data split and missing values**
df_full_train, df_test = train_test_split(df, test_size=0.2, random_state=1)

df_full_train = df_full_train.reset_index(drop=True)
df_test = df_test.reset_index(drop=True)

y_full_train = df_full_train.log_price.values
y_test = df_test.log_price.values

to_drop = ['price', 'log_price']
df_full_train = df_full_train.drop(columns=to_drop)
df_test = df_test.drop(columns=to_drop)

to_fill_median = df_full_train.columns[df_full_train.isnull().any()].tolist()
median_data = df_full_train[to_fill_median].median()
df_full_train = df_full_train.fillna(median_data)
df_test = df_test.fillna(median_data)

# ***MODEL***

train_dicts = df_full_train.to_dict(orient='records')
test_dicts = df_test.to_dict(orient='records')

dv = DictVectorizer(sparse=True)
X_train = dv.fit_transform(train_dicts)
X_test = dv.transform(test_dicts)
features = dv.get_feature_names_out().tolist()
dtrain = xgb.DMatrix(X_train, label=y_full_train,feature_names=features) 
dtest = xgb.DMatrix(X_test, label=y_test,feature_names=features)

xgb_params = {
    'eta': ETA,
    'max_depth': MAX_DEPTH,
    'min_child_weight': MIN_CHILD_WEIGHT,
    'objective': 'reg:squarederror',
    'nthread': 8,
    'seed': 1,
    'verbosity': 1,
}

xgb_final = xgb.train(xgb_params, dtrain, num_boost_round=NUM_BOOST_ROUND)
y_pred = xgb_final.predict(dtest)

rmse  = root_mean_squared_error(y_test, y_pred)
print('rmse test:', rmse)