#import pandas as pd
#import pickle

#from sklearn.model_selection import train_test_split
#from sklearn.model_selection import train_test_split
#from sklearn.ensemble import RandomForestClassifier
#from sklearn.preprocessing import LabelEncoder

# 1️⃣ Load dataset
#df = pd.read_csv("ecommerce.csv")

# 2️⃣ Drop unnecessary column
#df.drop("CustomerID", axis=1, inplace=True)

# 3️⃣ Handle missing values
#df.fillna(df.median(numeric_only=True), inplace=True)

# 4️⃣ Encode categorical columns
#cat_cols = [
    #"PreferredLoginDevice",
    #"PreferredPaymentMode",
   # "Gender",
  #  "PreferedOrderCat",
 #   "MaritalStatus"
#]

#encoder = LabelEncoder()
#for col in cat_cols:
 #   df[col] = encoder.fit_transform(df[col])

# 5️⃣ Features & Target
#X = df.drop("Churn", axis=1)
#y = df["Churn"]

# 6️⃣ Train-test split
#X_train, X_test, y_train, y_test = train_test_split(
 #   X, y, test_size=0.2, random_state=42
#)

# 7️⃣ Train Random Forest model
#model = RandomForestClassifier(
  #  n_estimators=100,
 #   random_state=42
#)

#model.fit(X_train, y_train)

# 8️⃣ Save model
#with open("model.pkl", "wb") as f:
 #   pickle.dump(model, f)

#print("✅ model.pkl successfully created")


import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

# Load dataset
df = pd.read_csv("ecommerce.csv")

# Select ONLY same features used in Streamlit
features = [
    "Tenure",
    "CityTier",
    "WarehouseToHome",
    "HourSpendOnApp",
    "NumberOfDeviceRegistered",
    "SatisfactionScore",
    "OrderCount",
    "CashbackAmount"
]

X = df[features]
y = df["Churn"]

# Handle missing values
X = X.fillna(X.median())

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train model
model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

model.fit(X_train, y_train)

# Save model
pickle.dump(model, open("model.pkl", "wb"))

print("✅ model.pkl recreated with 8 features")
