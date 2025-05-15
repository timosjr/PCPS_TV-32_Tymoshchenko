import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

model = joblib.load('pm25_model.pkl')

df = pd.read_csv('data.csv', parse_dates=['last_updated'])
df = df.dropna()

country = input("Введіть назву країни: ")

df_country = df[df['country'] == country].copy()

if df_country.empty:
    print("Дані для цієї країни не знайдено.")
else:
    df_country['month'] = df_country['last_updated'].dt.month

    plt.figure(figsize=(10, 5))
    sns.boxplot(x='month', y='air_quality_PM2.5', data=df_country)
    plt.title(f"Сезонна зміна PM2.5 у {country}")
    plt.xlabel("Місяць")
    plt.ylabel("PM2.5 (мкг/м³)")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    corr = df_country[['air_quality_PM2.5', 'temperature_celsius', 'humidity', 'wind_mph']].corr()
    print("\nКореляція між PM2.5 та погодними параметрами:")
    print(corr['air_quality_PM2.5'])

    latest = df_country[['temperature_celsius', 'humidity', 'wind_mph']].iloc[-1:, :]
    predicted_pm2_5 = model.predict(latest)[0]

    if predicted_pm2_5 <= 12:
        advice = "Добра якість повітря. Можна виходити."
    elif predicted_pm2_5 <= 35.4:
        advice = "Помірна якість. Краще бути обережним."
    else:
        advice = "Погана якість. Рекомендується залишитись вдома."

    print(f"\nПрогноз для {country}:")
    print(f"PM2.5 = {predicted_pm2_5:.2f} мкг/м³")
    print(f"Рекомендація: {advice}")

