import pandas as pd
from sklearn.model_selection import train_test_split
from utils_nans1 import *
import matplotlib.pyplot as plt


matplotlib.rcParams['figure.figsize'] = (8, 3)
sb.set(font_scale=1.)

df = pd.read_csv('./lungCancer.csv')
#print(df)
#print(df.isnull().sum())

df['GENDER'] = df['GENDER'].map({'M': 0, 'F': 1})
df['LUNG_CANCER'] = df['LUNG_CANCER'].map({'YES': 0, 'NO': 1})
print(df)

x = df.drop(columns=['LUNG_CANCER'])
y = df['LUNG_CANCER']

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, shuffle=True, random_state=42)
model = get_fitted_model(x_train, y_train)

r2 = get_rsquared_adj1(model, x_test, y_test)
print(r2)
print(model.summary())


from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Inicijalizacija i treniranje Random Forest klasifikatora
model_rf = RandomForestClassifier(random_state=42)
model_rf.fit(x_train, y_train)

# Predikcija na testnom skupu
predictions_rf = model_rf.predict(x_test)

# Evaluacija modela
accuracy_rf = accuracy_score(y_test, predictions_rf)
print(f"Random Forest Accuracy Score: {accuracy_rf}")



from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score

# Inicijalizacija i treniranje Gradient Boosting klasifikatora
gradient_boosting_model = GradientBoostingClassifier(random_state=42)
gradient_boosting_model.fit(x_train, y_train)

# Predikcija na testnom skupu
predictions_gb = gradient_boosting_model.predict(x_test)

# Evaluacija modela
accuracy_gb = accuracy_score(y_test, predictions_gb)
print(f"Gradient Boosting Accuracy Score: {accuracy_gb}")

