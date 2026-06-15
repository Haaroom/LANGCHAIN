import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler,OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
df = pd.read_csv("student_data.csv")
df = df.drop_duplicates()
target = "Risk"
X = df.drop(columns=[target])
y = df[target]
num_cols = X.select_dtypes(include=["int64", "float64"]).columns
cat_cols = X.select_dtypes(include=["object"]).columns
num_pipeline = Pipeline([("imputer", SimpleImputer(strategy="median")),("scaler", StandardScaler())])
cat_pipeline = Pipeline([("imputer", SimpleImputer(strategy="most_frequent")),("encoder", OneHotEncoder(handle_unknown="ignore"))])
preprocessor = ColumnTransformer([("num", num_pipeline, num_cols),("cat", cat_pipeline, cat_cols)])
pipeline = Pipeline([("preprocessor", preprocessor),("classifier", RandomForestClassifier(n_estimators=200,random_state=42))])
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=42,stratify=y)
pipeline.fit(X_train, y_train)
preds = pipeline.predict(X_test)
print(classification_report(y_test, preds))
joblib.dump(pipeline,"models/student_risk.pkl")
print("Model saved.")