import pandas as pd
import joblib
from sklearn.ensemble import RandomForestRegressor

df = pd.read_csv('data.csv')
df = df.dropna()

features = ['temperature_celsius', 'humidity', 'wind_mph']
X = df[features]
y = df['air_quality_PM2.5']

model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X, y)

joblib.dump(model, 'pm25_model.pkl')
print("Модель збережено у файл pm25_model.pkl")
