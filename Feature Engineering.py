
# coding: utf-8

# In[215]:


#Import the CSV Files
#house_train = Training Data
#house_test = Data on which the models needs to applied

df_train = pd.read_csv('data/house_train.csv') 
df_test = pd.read_csv('data/house_test.csv')


# In[216]:


#'describe' gives statistics of the data. In this case the training set
df_train['SalePrice'].describe()


# In[188]:


#'distplot' Flexibly plot a univariate distribution of observations.
sns.distplot(df_train['SalePrice']);


# This graph simply shows the statistics in an visualizing way. Since the mean is 180K, that is where the peak of the graph is. 

# In[189]:


#Print first 5 rows of the training dataset
df_train.head(5)


# You notice, a lot of the values are "NaN" = Not a number, meaning we have to take those into account. This means that the house is missing that data, which can effect the price of the house. So first we need to clean the data, before trying out any analysis. In our case we will using the "imputing" approach to fill the missing data with the means.
# 

# In[218]:


#This takes care of the Outliers, since they can affect the predicted modelling. 
df_train = df_train.drop(
    df_train[(df_train['GrLivArea']>4000) & (df_train['SalePrice']<300000)].index)


# In[219]:


# Concatenate data
all_data = pd.concat((df_train.loc[:,'MSSubClass':'SaleCondition'],
                      df_test.loc[:,'MSSubClass':'SaleCondition']))


# In[220]:


# Drop utilities column
all_data = all_data.drop(['Utilities'], axis=1)


# In[221]:


#Taken from kernels in Kaggle. This impute all the missing data
all_data["PoolQC"] = all_data["PoolQC"].fillna("None")
all_data["MiscFeature"] = all_data["MiscFeature"].fillna("None")
all_data["Alley"] = all_data["Alley"].fillna("None")
all_data["Fence"] = all_data["Fence"].fillna("None")
all_data["FireplaceQu"] = all_data["FireplaceQu"].fillna("None")
for col in ('GarageType', 'GarageFinish', 'GarageQual', 'GarageCond'):
    all_data[col] = all_data[col].fillna('None')
for col in ('BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2'):
    all_data[col] = all_data[col].fillna('None')
all_data['MSZoning'] = all_data['MSZoning'].fillna(all_data['MSZoning'].mode()[0])
all_data["MasVnrType"] = all_data["MasVnrType"].fillna("None")
all_data["Functional"] = all_data["Functional"].fillna("Typ")
all_data['Electrical'] = all_data['Electrical'].fillna(all_data['Electrical'].mode()[0])
all_data['KitchenQual'] = all_data['KitchenQual'].fillna(all_data['KitchenQual'].mode()[0])
all_data['Exterior1st'] = all_data['Exterior1st'].fillna(all_data['Exterior1st'].mode()[0])
all_data['Exterior2nd'] = all_data['Exterior2nd'].fillna(all_data['Exterior2nd'].mode()[0])
all_data['SaleType'] = all_data['SaleType'].fillna(all_data['SaleType'].mode()[0])
all_data['MSSubClass'] = all_data['MSSubClass'].fillna("None")


# In[222]:


#This is the best way to visually show the relations between variables
corrmat = df_train.corr()
f, ax = plt.subplots(figsize=(12, 9))
sns.heatmap(corrmat, vmax=.8, square=True);


# In[223]:


#This shows how the normal distribution works on the given graph.
sns.distplot(df_train['SalePrice'], fit=norm);
fig = plt.figure()
res = stats.probplot(df_train['SalePrice'], plot=plt)


# In[224]:


#Applying the log function
df_train['SalePrice'] = np.log(df_train['SalePrice'])


# In[225]:


sns.distplot(df_train['SalePrice'], fit=norm);
fig = plt.figure()
res = stats.probplot(df_train['SalePrice'], plot=plt)


# In[226]:


#Seeing how the log and GrLiveArea are related
sns.distplot(df_train['GrLivArea'], fit=norm);
fig = plt.figure()
res = stats.probplot(df_train['GrLivArea'], plot=plt)


# In[227]:


df_train['GrLivArea'] = np.log(df_train['GrLivArea'])


# In[228]:


sns.distplot(df_train['GrLivArea'], fit=norm);
fig = plt.figure()
res = stats.probplot(df_train['GrLivArea'], plot=plt)


# In[229]:


#Seeing how the TotalBsmtSF is related to teh prices
sns.distplot(df_train['TotalBsmtSF'], fit=norm);
fig = plt.figure()
res = stats.probplot(df_train['TotalBsmtSF'], plot=plt)


# In[230]:


#This fills the missing data with the median of all neighborhood
all_data["LotFrontage"] = all_data.groupby("Neighborhood")["LotFrontage"].transform(
    lambda x: x.fillna(x.median()))

for col in ('GarageYrBlt', 'GarageArea', 'GarageCars'):
    all_data[col] = all_data[col].fillna(0)
    
for col in ('BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF','TotalBsmtSF', 'BsmtFullBath', 'BsmtHalfBath'):
    all_data[col] = all_data[col].fillna(0)
    
all_data["MasVnrArea"] = all_data["MasVnrArea"].fillna(0)


# In[231]:


#Since some data is in numerics which doesn't have any affect in numbers, it needs to be converted to categorical data

#MSSubClass=The building class
all_data['MSSubClass'] = all_data['MSSubClass'].apply(str)

#Changing OverallCond into a categorical variable
all_data['OverallCond'] = all_data['OverallCond'].astype(str)

#Year and month sold are transformed into categorical features.
all_data['YrSold'] = all_data['YrSold'].astype(str)
all_data['MoSold'] = all_data['MoSold'].astype(str)


# In[232]:


#Total square footage needs to be added to the data, since it correlates directly to the houseprice
all_data['TotalSF'] = all_data['TotalBsmtSF'] + all_data['1stFlrSF'] + all_data['2ndFlrSF']


# In[233]:


#Adding total sqfootage feature 
all_data['TotalSF'] = all_data['TotalBsmtSF'] + all_data['1stFlrSF'] + all_data['2ndFlrSF']

#log transform skewed numeric features:
numeric_feats = all_data.dtypes[all_data.dtypes != "object"].index

skewed_feats = all_data[numeric_feats].apply(lambda x: skew(x.dropna())) #compute skewness
skewed_feats = skewed_feats[skewed_feats > 0.75]
skewed_feats = skewed_feats.index

all_data[skewed_feats] = np.log1p(all_data[skewed_feats])


# In[234]:


cols = ('FireplaceQu', 'BsmtQual', 'BsmtCond', 'GarageQual', 'GarageCond', 
        'ExterQual', 'ExterCond','HeatingQC', 'PoolQC', 'KitchenQual', 'BsmtFinType1', 
        'BsmtFinType2', 'Functional', 'Fence', 'BsmtExposure', 'GarageFinish', 'LandSlope',
        'LotShape', 'PavedDrive', 'Street', 'Alley', 'CentralAir', 'MSSubClass', 'OverallCond', 
        'YrSold', 'MoSold')

# process columns, apply LabelEncoder to categorical features
for c in cols:
    lbl = LabelEncoder() 
    lbl.fit(list(all_data[c].values)) 
    all_data[c] = lbl.transform(list(all_data[c].values))
    
all_data = pd.get_dummies(all_data)


# In[235]:


X_train = all_data[:df_train.shape[0]]
X_test = all_data[df_train.shape[0]:]

y = df_train.SalePrice


# In[236]:


# Initialize models
lr = LinearRegression(
    n_jobs = -1
)

rd = Ridge(
    alpha = 4.84
)

rf = RandomForestRegressor(
    n_estimators = 12,
    max_depth = 3,
    n_jobs = -1
)

gb = GradientBoostingRegressor(
    n_estimators = 40,
    max_depth = 2
)

nn = MLPRegressor(
    hidden_layer_sizes = (90, 90),
    alpha = 2.75
)
# Initialize Ensemble
model = StackingRegressor(
    regressors=[rf, gb, nn, rd],
    meta_regressor=lr
)

# Fit the model on our data
model.fit(X_train, y)


# In[237]:


# Predict training set
y_pred = model.predict(X_train)
print(sqrt(mean_squared_error(y, y_pred)))


# In[238]:


Y_pred = model.predict(X_test)


# In[239]:


# Create empty submission dataframe
sub = pd.DataFrame()

# Insert ID and Predictions into dataframe
sub['Id'] = df_test['Id']
sub['SalePrice'] = np.expm1(Y_pred)

# Output submission file
sub.to_csv('data/sample_submission.csv',index=False)


# In[ ]:


#Plenty of notebooks from Kaggles were used to create this.

