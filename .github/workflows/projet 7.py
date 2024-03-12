#!/usr/bin/env python
# coding: utf-8

# In[1]:


import mlflow
from mlflow.models import infer_signature


# In[2]:


mlflow.set_tracking_uri(uri="http://127.0.0.1:5000")


# In[3]:


experiment_name = "/Shared/MLflowP7/"


# In[4]:


mlflow.set_experiment(experiment_name)


# In[5]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


# In[6]:


application_test = pd.read_csv("application_test.csv")
application_train = pd.read_csv("application_train.csv")
bureau = pd.read_csv("bureau[1].csv")
bureau_balance = pd.read_csv("bureau_balance.csv")
credit_card_balance = pd.read_csv("credit_card_balance.csv")
HomeCredit_columns_description = pd.read_csv("HomeCredit_columns_description.csv", encoding='iso-8859-1')
installments_payments = pd.read_csv("installments_payments[1].csv")
POS_CASH_balance = pd.read_csv("POS_CASH_balance.csv")
previous_application = pd.read_csv("previous_application.csv")
sample_submission = pd.read_csv("sample_submission[1].csv")


# In[7]:


column_mapping = dict(zip(HomeCredit_columns_description['Row'], HomeCredit_columns_description['Description']))


# In[8]:


application_train_2 = application_train.rename(columns=column_mapping)


# In[9]:


application_test_2 = application_test.rename(columns=column_mapping)


# In[10]:


application_train.shape 


# In[11]:


application_train.dtypes.value_counts()


# In[12]:


def missing(df):
    missing_data = pd.DataFrame(round(df.isna().sum()*100/len(df),1)).reset_index().rename(columns={"index":"column",0:"missing_data"})
    return missing_data


# In[13]:


missing(application_train)


# In[14]:


plt.figure(figsize=(12,24))
sns.barplot(x="missing_data", y="column", data=missing(application_train), color="skyblue")
plt.plot([40,40],[0,125], color="red")


# In[15]:


application_train.select_dtypes(include=["float64"]).shape


# In[16]:


application_train.select_dtypes(include=["int"]).shape


# In[17]:


application_train.select_dtypes(include=["object"]).shape


# In[18]:


missing(application_train.select_dtypes(include=["float64"]))


# # FLOAT VARIABLES 

# In[19]:


plt.figure(figsize=(8,12))
sns.barplot(x="missing_data", y="column", data=missing(application_train.select_dtypes(include=["float64"])), color="skyblue")
plt.plot([40,40],[0,65], color="red")


# In[20]:


print(missing(application_train.select_dtypes(include=["float64"]))[missing(application_train.select_dtypes(include=["float64"])).missing_data>40].shape[0],'/',len(missing(application_train.select_dtypes(include=["float64"]))))


# In[21]:


def corr(df,typ,miss,cor):
    dataframe = df[missing(df.select_dtypes(include=[typ]))[missing(df.select_dtypes(include=[typ])).missing_data>=miss].column.values].corr()
    dataframe = dataframe[dataframe>cor].count()
    dataframe = dataframe[dataframe>1]
    plt.figure(figsize=(22,16))
    heatmap = list(dataframe.index)
    heatmap = sns.heatmap(df.select_dtypes(include=[typ])[df.select_dtypes(include=[typ]).columns.intersection(heatmap)].corr(),annot=True)
    return dataframe, heatmap


# In[22]:


corr(application_train,'float64',40,0.6)


# In[23]:


mode_medi_drop = [x for x in application_train.select_dtypes(include=["float64"]) if "MODE" in x or "MEDI" in x]
application_train_float = application_train.select_dtypes(include=["float64"]).drop(columns=mode_medi_drop)


# In[24]:


application_train_float.shape


# In[25]:


application_train.shape


# In[26]:


plt.figure(figsize=(8,12))
sns.barplot(x="missing_data", y="column", data=missing(application_train_float), color="skyblue")
plt.plot([40,40],[0,35], color="red")


# In[27]:


corr(application_train_float,'float64',40,0.6)


# In[28]:


apartments_drop = [x for x in application_train_float.select_dtypes(include=["float64"]) if "LIVING" in x or "FLOOR" in x or "BASE" in x or "ENT" in x or "ELE" in x]
application_train_float = application_train_float.select_dtypes(include=["float64"]).drop(columns=apartments_drop)


# In[29]:


application_train.shape


# In[30]:


corr(application_train_float,'float64',0,0)


# In[31]:


last_drop = [x for x in application_train_float.select_dtypes(include=["float64"]) if "DEF_30" in x or "AMT_GOODS" in x or "AMT_ANNUITY" in x]
application_train_float = application_train_float.select_dtypes(include=["float64"]).drop(columns=last_drop)


# In[32]:


application_train.shape


# In[33]:


application_train_float


# In[34]:


plt.figure(figsize=(8,12))
sns.barplot(x="missing_data", y="column", data=missing(application_train_float), color="skyblue")
plt.plot([40,40],[0,25], color="red")


# # INT VARIABLES

# In[35]:


application_train.select_dtypes(include=["int"])


# In[36]:


plt.figure(figsize=(8,12))
sns.barplot(x="missing_data", y="column", data=missing(application_train.select_dtypes(include=["int"])), color="skyblue")


# In[37]:


corr(application_train,'int',0,0.6)


# In[38]:


drop_int = [x for x in application_train.select_dtypes(include=["int"]) if "REG_REGION_NOT_WORK" in x or "REG_CITY_NOT_WORK" in x or "REGION_RATING_CLIENT_W_CITY" in x or "FLAG_EMP_PHONE" in x]
application_train_int = application_train.select_dtypes(include=["int"]).drop(columns=drop_int)


# In[39]:


application_train_int.shape


# # OBJECT VARIABLES

# In[40]:


missing(application_train.select_dtypes(include=["object"]))


# In[41]:


plt.figure(figsize=(8,8))
sns.barplot(x="missing_data", y="column", data=missing(application_train.select_dtypes(include=["object"])), color="skyblue")
plt.plot([40,40],[0,17], color="red")


# In[42]:


HomeCredit_columns_description[HomeCredit_columns_description['Row'] == "FONDKAPREMONT_MODE"]


# In[43]:


application_train_obj = application_train.select_dtypes(include=["object"])


# In[44]:


application_train_obj.shape


# # Encode OBJ

# In[45]:


from sklearn.preprocessing import LabelEncoder


# In[46]:


application_train_obj.nunique()


# In[47]:


le = LabelEncoder()


# In[48]:


columns_object_impute = ["NAME_TYPE_SUITE","OCCUPATION_TYPE","FONDKAPREMONT_MODE","HOUSETYPE_MODE","WALLSMATERIAL_MODE","EMERGENCYSTATE_MODE"]


# In[49]:


application_train_obj[columns_object_impute]


# In[50]:


application_train_obj[columns_object_impute].apply(LabelEncoder().fit_transform)


# In[51]:


sns.heatmap(application_train_obj[columns_object_impute].apply(LabelEncoder().fit_transform).corr(),annot=True)


# In[52]:


drop_obj = [ x for x in application_train.select_dtypes(include=["object"]) if "FONDKAPREMONT_MODE" in x or "WALLSMATERIAL" in x or "EMERGENCYSTATE_MODE" in x]
application_train_obj = application_train.select_dtypes(include=["object"]).drop(columns=drop_obj)


# In[53]:


application_train_obj.shape


# In[54]:


application_train_obj


# In[55]:


final_df_1 = application_train_int.join(application_train_float)
final_df = final_df_1.join(application_train_obj)


# In[56]:


final_df


# In[57]:


plt.figure(figsize=(8,12))
sns.barplot(x="missing_data", y="column", data=missing(final_df), color="skyblue")
plt.plot([40,40],[0,75], color="red")


# # Missing data imputation

# ## categorical data

# In[58]:


from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.preprocessing import OrdinalEncoder


# In[59]:


numerical_cols1 = final_df.select_dtypes(include=['int64', 'float64']).columns
categorical_cols1 = final_df.select_dtypes(include=['object']).columns


# In[60]:


numerical_data1 = final_df[numerical_cols1]
categorical_data1 = final_df[categorical_cols1]


# In[61]:


encoder = OrdinalEncoder()


# In[62]:


categorical_data1.fillna('Missing', inplace=True)


# In[63]:


categorical_data_encoded1 = encoder.fit_transform(categorical_data1)


# In[64]:


categorical_data_encoded1 = pd.DataFrame(categorical_data_encoded1, columns=categorical_cols1)


# ## numerical data

# In[65]:


imputer = IterativeImputer(max_iter=10, random_state=0)
numerical_data_imputed1 = imputer.fit_transform(numerical_data1)
numerical_data_imputed1 = pd.DataFrame(numerical_data_imputed1, columns=numerical_cols1)


# In[66]:


final_df = pd.concat([numerical_data_imputed1, categorical_data_encoded1], axis=1)


# # Important features 

# ## Gradient Boosting regression

# In[67]:


from sklearn import datasets, ensemble
from sklearn.inspection import permutation_importance
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split


# In[68]:


X, y = final_df.drop(columns="TARGET"), final_df["TARGET"]


# In[69]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=13)
params1 = {"n_estimators":100, "learning_rate":1.0, "max_depth":1, "random_state":0}


# In[70]:


clf1 = ensemble.GradientBoostingClassifier(**params1).fit(X_train, y_train)
clf1.score(X_test, y_test)


# In[71]:


feature_importances1 = clf1.feature_importances_


# In[72]:


indices1 = feature_importances1.argsort()[::-1]


# In[73]:


plt.figure(figsize=(22, 8))
plt.title("Feature Importances")
plt.bar(range(X_train.shape[1]), feature_importances1[indices1], color="r", align="center")
plt.xticks(range(X_train.shape[1]), indices1)
plt.xlim([-1, X_train.shape[1]])
plt.xlabel('Feature Index')
plt.ylabel('Importance')
plt.axvline(x=4.5, color='blue', linewidth=2, linestyle='--')
plt.show()


# In[74]:


indices1_5 = indices1[:5]


# In[75]:


final_df.columns[indices1_5]


# In[76]:


numerical_cols2 = application_train.select_dtypes(include=['int64', 'float64']).columns
categorical_cols2 = application_train.select_dtypes(include=['object']).columns
numerical_data2 = application_train[numerical_cols2]
categorical_data2 = application_train[categorical_cols2]


# In[77]:


categorical_data2.fillna('Missing', inplace=True)
categorical_data_encoded2 = encoder.fit_transform(categorical_data2)
categorical_data_encoded2 = pd.DataFrame(categorical_data_encoded2, columns=categorical_cols2)


# In[78]:


numerical_data_imputed2 = imputer.fit_transform(numerical_data2)
numerical_data_imputed2 = pd.DataFrame(numerical_data_imputed2, columns=numerical_cols2)


# In[79]:


application_train_brut = pd.concat([numerical_data_imputed2, categorical_data_encoded2], axis=1)


# In[80]:


X, y = application_train_brut.drop(columns="TARGET"), application_train_brut["TARGET"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=13)
params2 = {"n_estimators":100, "learning_rate":1.0, "max_depth":1, "random_state":0}
clf2 = ensemble.GradientBoostingClassifier(**params2).fit(X_train, y_train)
clf2.score(X_test, y_test)
feature_importances2 = clf2.feature_importances_


# In[81]:


clf2.score(X_test, y_test)


# In[82]:


indices2 = feature_importances2.argsort()[::-1]


# In[83]:


plt.figure(figsize=(22, 8))
plt.title("Feature Importances")
plt.bar(range(X_train.shape[1]), feature_importances2[indices2], color="r", align="center")
plt.xticks(range(X_train.shape[1]), indices2)
plt.xlim([-1, X_train.shape[1]])
plt.xlabel('Feature Index')
plt.ylabel('Importance')
plt.axvline(x=4.5, color='blue', linewidth=2, linestyle='--')
plt.show()


# In[84]:


indices2_5 = indices2[:5]


# In[85]:


application_train_brut.columns[indices2_5]


# In[86]:


feat2 = pd.DataFrame(application_train_brut.columns[indices2_5])
feat2.rename(columns={0 :'imp_feat2_all'}, inplace=True)
feat1 = pd.DataFrame(final_df.columns[indices1_5])
feat1.rename(columns={0 :'imp_feat1_corr_remove'}, inplace=True)


# In[87]:


featmix1 = pd.DataFrame(application_train_brut.columns[indices2_5][:6])
featmix1.rename(columns={0 :'imp_featmix'}, inplace=True)
featmix2 = pd.DataFrame(final_df.columns[indices1_5][:5])
featmix2.rename(columns={0 :'imp_featmix'}, inplace=True)
featmix = [featmix1,featmix2]
featmix = pd.concat(featmix).drop_duplicates().reset_index(drop=True)


# In[88]:


imp_feat = feat2.join(feat1)


# In[89]:


imp_feat = imp_feat.join(featmix)


# In[90]:


imp_feat


# In[91]:


HomeCredit_columns_description[HomeCredit_columns_description.Row == "EXT_SOURCE_2"].Description # Name of the most important feature 


# ### The most important feature is the EXT_SOURCE_2 which corresponds to a Normalized score from external data source, the next features are less lower than the first one, we will incorporate the first 5 features in the models training.

# In[92]:


featmix_5 = imp_feat.imp_featmix.tolist()


# ## Performance test of the 3 datasets

# In[93]:


from sklearn.metrics import precision_score, recall_score, roc_auc_score, roc_curve


# In[94]:


final_df_5 = final_df[final_df.columns[indices1_5]]


# In[95]:


final_df_5 = final_df[["TARGET"]].join(final_df_5)


# In[96]:


application_train_brut_5 = application_train_brut[application_train_brut.columns[indices2_5]]


# In[97]:


application_train_brut_5 = application_train_brut[["TARGET"]].join(application_train_brut_5)


# In[98]:


datafeatmix_5 = application_train_brut[featmix_5]


# In[99]:


datafeatmix_5 = application_train_brut[["TARGET"]].join(datafeatmix_5)


# In[100]:


X1, y1 = final_df_5.drop(columns="TARGET"), final_df_5["TARGET"]
X_train1, X_test1, y_train1, y_test1 = train_test_split(X1, y1, test_size=0.2, random_state=0)
params1 = {"n_estimators":100, "learning_rate":1.0, "max_depth":1, "random_state":0}
clf1 = ensemble.GradientBoostingClassifier(**params1).fit(X_train1, y_train1)


# In[101]:


X2, y2 = application_train_brut_5.drop(columns="TARGET"), application_train_brut_5["TARGET"]
X_train2, X_test2, y_train2, y_test2 = train_test_split(X2, y2, test_size=0.2, random_state=0)
params2 = {"n_estimators":100, "learning_rate":1.0, "max_depth":1, "random_state":0}
clf2 = ensemble.GradientBoostingClassifier(**params2).fit(X_train2, y_train2)


# In[102]:


X3, y3 = datafeatmix_5.drop(columns="TARGET"), datafeatmix_5["TARGET"]
X_train3, X_test3, y_train3, y_test3 = train_test_split(X3, y3, test_size=0.2, random_state=0)
params3 = {"n_estimators":100, "learning_rate":1.0, "max_depth":1, "random_state":0}
clf3 = ensemble.GradientBoostingClassifier(**params3).fit(X_train3, y_train3)


# In[103]:


y_pred1 = clf1.predict(X_test1)
y_pred_proba1 = clf1.predict_proba(X_test1)[:, 1]  


# In[104]:


y_pred2 = clf2.predict(X_test2)
y_pred_proba2 = clf2.predict_proba(X_test2)[:, 1] 


# In[105]:


y_pred3 = clf3.predict(X_test3)
y_pred_proba3 = clf3.predict_proba(X_test3)[:, 1] 


# In[106]:


precision1 = precision_score(y_test1, y_pred1)
recall1 = recall_score(y_test1, y_pred1)


# In[107]:


precision2 = precision_score(y_test2, y_pred2)
recall2 = recall_score(y_test2, y_pred2)


# In[108]:


precision3 = precision_score(y_test3, y_pred3)
recall3 = recall_score(y_test3, y_pred3)


# In[109]:


auc1 = roc_auc_score(y_test1, y_pred_proba1)
auc2 = roc_auc_score(y_test2, y_pred_proba2)
auc3 = roc_auc_score(y_test3, y_pred_proba3)


# In[110]:


print(f"Precision: {precision1:.2f}")
print(f"Recall: {recall1:.2f}")
print(f"AUC: {auc1:.2f}")


# In[111]:


print(f"Precision: {precision2:.2f}")
print(f"Recall: {recall2:.2f}")
print(f"AUC: {auc2:.2f}")


# In[112]:


print(f"Precision: {precision3:.2f}")
print(f"Recall: {recall3:.2f}")
print(f"AUC: {auc3:.2f}")


# In[113]:


fpr, tpr, thresholds = roc_curve(y_test1, y_pred_proba1)
plt.plot(fpr, tpr, label=f'ROC curve (area = {auc1:.2f})')
plt.plot([0, 1], [0, 1], 'k--')  # Ligne diagonale pour un modèle aléatoire
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) for dataset treated for multicollinearity ')
plt.legend(loc="lower right")
plt.show()


# In[114]:


fpr, tpr, thresholds = roc_curve(y_test2, y_pred_proba2)
plt.plot(fpr, tpr, label=f'ROC curve (area = {auc2:.2f})')
plt.plot([0, 1], [0, 1], 'k--')  # Ligne diagonale pour un modèle aléatoire
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) for dataset selected with gradient boost classifier')
plt.legend(loc="lower right")
plt.show()


# In[115]:


fpr, tpr, thresholds = roc_curve(y_test3, y_pred_proba3)
plt.plot(fpr, tpr, label=f'ROC curve (area = {auc3:.2f})')
plt.plot([0, 1], [0, 1], 'k--')  # Ligne diagonale pour un modèle aléatoire
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) for dataset selected by using both techniques')
plt.legend(loc="lower right")
plt.show()


# ### The selected dataset is based on the most important features selected by using Gradient boosting Classifier after removing collinear features 

# # Distribution of target variable

# In[116]:


# Distribution of the resampled target variable 
value_counts = final_df_5['TARGET'].value_counts(normalize=True) * 100
ax = sns.countplot(x="TARGET", data=datafeatmix_5, order=value_counts.index)
plt.title("Distribution of Target")
for p in ax.patches:
    percentage = '{:.1f}%'.format(100 * p.get_height() / len(final_df_5))
    x = p.get_x() + p.get_width() / 1.7 - 0.05
    y = p.get_y() + p.get_height() / 2.7
    ax.annotate(percentage, (x, y), ha='center')

# Show the plot
plt.show()


# ### The dataset is very unbalanced therefore we will resample the dataset to maximize the the lower Target category=1

# In[117]:


from imblearn.over_sampling import SMOTE


# In[118]:


sm = SMOTE(random_state=0)


# In[119]:


resampled_X, y_resampled = sm.fit_resample(final_df_5.drop(columns="TARGET"), final_df_5["TARGET"])


# In[120]:


resampled_df = pd.DataFrame(y_resampled).join(resampled_X)


# In[121]:


resampled_df


# In[122]:


# Distribution of the resampled target variable 
value_counts = resampled_df['TARGET'].value_counts(normalize=True) * 100
ax = sns.countplot(x="TARGET", data=resampled_df, order=value_counts.index)
plt.title("Distribution of Target")
for p in ax.patches:
    percentage = '{:.1f}%'.format(100 * p.get_height() / len(final_df_5))
    x = p.get_x() + p.get_width() / 1.7 - 0.05
    y = p.get_y() + p.get_height() / 2.7
    ax.annotate(percentage, (x, y), ha='center')

# Show the plot
plt.show()


# In[123]:


# Define the business cost function
def business_cost(y_true, y_pred, cost_fn, cost_fp):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    return (cost_fn * fn) + (cost_fp * fp)


# In[124]:


# Define the function to optimize the threshold
def optimize_threshold(y_true, y_pred_proba, cost_fn, cost_fp):
    thresholds = np.linspace(0, 1, 100)
    costs = []
    for threshold in thresholds:
        y_pred = (y_pred_proba[:, 1] >= threshold).astype(int)
        costs.append(business_cost(y_true, y_pred, cost_fn, cost_fp))
    optimal_threshold = thresholds[np.argmin(costs)]
    return optimal_threshold, min(costs)


# In[125]:


from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import confusion_matrix


# In[126]:


scaler = StandardScaler()


# In[127]:


X_train, X_test, y_train, y_test = train_test_split(resampled_df.drop(columns=['TARGET']), resampled_df['TARGET'],train_size=0.95)


# In[128]:


X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


# # Gradient Boosting Classifier

# In[129]:


with mlflow.start_run():
    params = {"n_estimators":[100,200], "learning_rate":[0.1,0.5], "max_depth":[3,5]}
    # Create and train model
    GBC = ensemble.GradientBoostingClassifier()

    gsvgbc = GridSearchCV(GBC, params, cv=5)
    gsvgbc.fit(X_train_scaled, y_train)
    
    mlflow.log_params(gsvgbc.best_params_)
    
    pipeline_gbc = Pipeline([('scaler', StandardScaler()), 
                              ('GBC', ensemble.GradientBoostingClassifier(**gsvgbc.best_params_))])
    
    pipeline_gbc.fit(X_train, y_train)
    
    y_pred_proba = pipeline_gbc.predict_proba(final_df_5.drop(columns="TARGET"))
    
    # Optimize the threshold based on the business cost
    cost_fn = 10  # Cost of a false negative
    cost_fp = 1   # Cost of a false positive
    optimal_threshold, min_cost = optimize_threshold(final_df_5["TARGET"], y_pred_proba, cost_fn, cost_fp)
    
    # Log the optimal threshold and minimum cost
    mlflow.log_metric("optimal_threshold GBC", optimal_threshold)
    mlflow.log_metric("min_cost GBC", min_cost)
    
    y_pred_adjusted = (y_pred_proba[:,1] >= optimal_threshold).astype(int)
    
    # Create metrics
    precision = precision_score(final_df_5["TARGET"], y_pred_adjusted)
    recall = recall_score(final_df_5["TARGET"], y_pred_adjusted)
    auc = roc_auc_score(final_df_5["TARGET"], y_pred_proba[:,1])
    
    # Log metrics
    mlflow.log_metric("precision GBC", precision)
    mlflow.log_metric("recall GBC", recall)
    mlflow.log_metric("auc GBC", auc)


# # K Neighbors Classifier

# In[130]:


from sklearn.neighbors import KNeighborsClassifier
with mlflow.start_run():
    params = {"n_neighbors":[3,5], 'weights':['uniform', 'distance'], 'metric':['euclidean','minkowski']}
    # Create and train model
    model = KNeighborsClassifier()

    gsvknn = GridSearchCV(model, params, cv=5)
    gsvknn.fit(X_train_scaled, y_train)
    
    mlflow.log_params(gsvknn.best_params_)
    
    pipeline_knn = Pipeline([('scaler', StandardScaler()), 
                              ('model', KNeighborsClassifier(**gsvknn.best_params_))])
    
    pipeline_knn.fit(X_train, y_train)
    
    y_pred_proba = pipeline_knn.predict_proba(final_df_5.drop(columns="TARGET"))
    
    # Optimize the threshold based on the business cost
    cost_fn = 10  # Cost of a false negative
    cost_fp = 1   # Cost of a false positive
    optimal_threshold, min_cost = optimize_threshold(final_df_5["TARGET"], y_pred_proba, cost_fn, cost_fp)
    
    # Log the optimal threshold and minimum cost
    mlflow.log_metric("optimal_threshold KNN", optimal_threshold)
    mlflow.log_metric("min_cost KNN", min_cost)
    
    y_pred_adjusted = (y_pred_proba[:,1] >= optimal_threshold).astype(int)
    
    # Create metrics
    precision = precision_score(final_df_5["TARGET"], y_pred_adjusted)
    recall = recall_score(final_df_5["TARGET"], y_pred_adjusted)
    auc = roc_auc_score(final_df_5["TARGET"], y_pred_proba[:,1])
    
    # Log metrics
    mlflow.log_metric("precision KNN", precision)
    mlflow.log_metric("recall KNN", recall)
    mlflow.log_metric("auc KNN", auc)


# # Logistic Regression

# In[131]:


from sklearn.linear_model import LogisticRegression
with mlflow.start_run():
    params = {"penalty":['l1', 'l2', None],"C":[0.5, 1.0, 1.5], 'solver':['lbfgs', 'liblinear', 'newton-cg']}
    # Create and train model
    model = LogisticRegression()

    gsvlr = GridSearchCV(model, params, cv=5)
    gsvlr.fit(X_train_scaled, y_train)
    
    mlflow.log_params(gsvlr.best_params_)
    
    pipeline_lr = Pipeline([('scaler', StandardScaler()), 
                              ('model', LogisticRegression(**gsvlr.best_params_))])
    
    pipeline_lr.fit(X_train, y_train)
    
    y_pred_proba = pipeline_lr.predict_proba(final_df_5.drop(columns="TARGET"))
    
    # Optimize the threshold based on the business cost
    cost_fn = 10  # Cost of a false negative
    cost_fp = 1   # Cost of a false positive
    optimal_threshold, min_cost = optimize_threshold(final_df_5["TARGET"], y_pred_proba, cost_fn, cost_fp)
    
    # Log the optimal threshold and minimum cost
    mlflow.log_metric("optimal_threshold LR", optimal_threshold)
    mlflow.log_metric("min_cost LR", min_cost)
    
    y_pred_adjusted = (y_pred_proba[:,1] >= optimal_threshold).astype(int)
    
    # Create metrics
    precision = precision_score(final_df_5["TARGET"], y_pred_adjusted)
    recall = recall_score(final_df_5["TARGET"], y_pred_adjusted)
    auc = roc_auc_score(final_df_5["TARGET"], y_pred_proba[:,1])
    
    # Log metrics
    mlflow.log_metric("precision LR", precision)
    mlflow.log_metric("recall LR", recall)
    mlflow.log_metric("auc LR", auc)


# ### Le modèle choisit est le KNN C 

# In[132]:


# Training the model with the all the dataset


# In[133]:


X_resampled_df, y_resampled_df = resampled_df.drop(columns=['TARGET']), pd.DataFrame(resampled_df['TARGET'])


# In[134]:


X_train, X_test, y_train, y_test = train_test_split(resampled_df.drop(columns=['TARGET']), resampled_df['TARGET'],train_size=0.95)


# In[135]:


with mlflow.start_run():
    # Create and train model
    pipeline_knn.fit(X_resampled_df, y_resampled_df)
    
    y_pred = pipeline_knn.predict(final_df_5.drop(columns="TARGET"))
    y_pred_proba = pipeline_knn.predict_proba(final_df_5.drop(columns="TARGET"))
     
    # Create metrics
    precision = precision_score(final_df_5["TARGET"], y_pred)
    recall = recall_score(final_df_5["TARGET"], y_pred)
    auc = roc_auc_score(final_df_5["TARGET"], y_pred_proba[:,1])
    
    # Log metrics
    mlflow.log_metric("precision KNN_ALL", precision)
    mlflow.log_metric("recall KNN_ALL", recall)
    mlflow.log_metric("auc KNN_ALL", auc)


# In[136]:


import pickle


# In[137]:


filename = 'creditmodel.sav'


# In[138]:


with open(filename, 'wb') as file:
    pickle.dump(pipeline_knn, file) # saving the selected model which is the KNN with best parameters  


# In[139]:


import os 


# In[140]:


model_path = os.path.abspath(filename)


# In[141]:


model_path


# In[142]:


pickled_model = pickle.load(open(filename, 'rb'))


# In[143]:


final_df_5.columns


# In[144]:


# Name of the application features 
print(list(HomeCredit_columns_description[HomeCredit_columns_description.Row == "EXT_SOURCE_2"].Description))  
print(list(HomeCredit_columns_description[HomeCredit_columns_description.Row == "EXT_SOURCE_3"].Description))  
print(list(HomeCredit_columns_description[HomeCredit_columns_description.Row == "CNT_FAM_MEMBERS"].Description))  
print(list(HomeCredit_columns_description[HomeCredit_columns_description.Row == "DAYS_REGISTRATION"].Description))
print(list(HomeCredit_columns_description[HomeCredit_columns_description.Row == "AMT_REQ_CREDIT_BUREAU_HOUR"].Description))  


# In[ ]:





# In[ ]:





# In[ ]:




