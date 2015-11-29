import pandas as pd
import statsmodels.api as sm
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.cross_validation import train_test_split
from sklearn.cross_validation import cross_val_score

pd.set_option('display.width', 120)
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 50)
pd.set_option('display.float_format', lambda x: '%.3f' % x)
np.set_printoptions(suppress=True, threshold = 'nan')

#########################
##     STATSMODELS     ##
#########################

# To what extent do specific factors contribute to death, among drivers only? (ML)

ml = pd.read_pickle('fars.pkl')
ml = ml[ml['PER_TYP'] == 'Driver of a Motor Vehicle In-Transport']

""" FEATURE CLEANING """
def prune(ml, feature, n):
    """ This function keeps the top n features and sets the rest to 'Other' """
    rel_vals = list(ml[feature].value_counts().head(n).index.values)
    def prune_features(df):
        if df[feature] not in rel_vals:
            df[feature] = 'Other'
        return df
    ml = ml.apply(prune_features, axis=1)
    return ml

categorical_features = ['SEX', 'INJ_SEV', 'REST_USE', 'VSURCOND', 'DEFORMED', 'WEATHER', 'LGT_COND', 'RELJCT2', 'BODY_SIZE']
for x, n in zip(categorical_features, [2, 4, 5, 2, 3, 5, 4, 4, 5, 6]):
    ml = prune(ml, x, n)

""" INJ_SEV IS THE DEPENDENT VARIABLE """
categorical_features.extend(['DR_DRINK', 'DR_SF', 'DRUG_DUMMY', 'SPEED_DUMMY'])
categorical_dummies = pd.get_dummies(ml[categorical_features])
continuous_features = ['AGE', 'NUMOCCS', 'MOD_YEAR', 'TRAV_SP', 'DR_HGT', 'DR_WGT', 'PREV_SUS', 'PREV_DWI', 'PREV_SPD']

# drop variables to avoid perfect separation: when one or more explanatory variables perfectly explains variation in the dependent variable
unrelated = ['SEX_Other', 'INJ_SEV_No Apparent Injury', 'INJ_SEV_Other', 'INJ_SEV_Suspected Minor Injury', 'INJ_SEV_Suspected Serious Injury', 'DEFORMED_Other', 'LGT_COND_Other', 'RELJCT2_Other']
reference = ['SEX_Female', 'REST_USE_None Used', 'VSURCOND_Dry', 'DEFORMED_Minor Damage', 'WEATHER_Clear', 'LGT_COND_Daylight', 'RELJCT2_Non-Junction', 'DR_DRINK_No Drinking', 'BODY_SIZE_Small']

ml_recoded = pd.concat([ml[continuous_features], categorical_dummies], axis=1)
ml_recoded = ml_recoded.drop(unrelated, axis=1)
ml_recoded = ml_recoded.drop(reference, axis=1)
ml_recoded = ml_recoded.fillna(ml_recoded.mean())

X = ml_recoded.drop('INJ_SEV_Fatal Injury', axis=1)
X = sm.tools.tools.add_constant(X)
y = ml_recoded['INJ_SEV_Fatal Injury']
logit = sm.Logit(y, X)
result = logit.fit()
print result.summary()

data = {k: np.exp(v) for k, v in result.params.iteritems()}
odds = pd.DataFrame(data.items(), columns=['Variable', 'Odds']).sort('Variable')
print odds

########################
##     CLASSIFIER     ##
########################

X, y = X.values, y.values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=35)
LR = LogisticRegression()
LR = LR.fit(X_train, y_train)
LR.score(X_test, y_test)

# test for overfitting
scores = cross_val_score(LogisticRegression(), X, y, scoring='accuracy', cv=10)
print scores.mean()

LR.predict(X_test[0:10])

