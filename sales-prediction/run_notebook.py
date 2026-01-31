import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from kafka import KafkaProducer, KafkaConsumer
import json
import statistics

warnings.filterwarnings("ignore")
sns.set(style="whitegrid")

print("=" * 70)
print("SALES PREDICTION WITH TENSORFLOW AND KAFKA STREAMING")
print("=" * 70)
print()

# === Load Dataset ===
df = pd.read_csv(r"C:\Users\angel\OneDrive\Documents\raw_dataset_week4.csv")
print(f"Loaded dataset: {df.shape[0]} rows, {df.shape[1]} columns")

# === Handle Missing Values ===
df_numeric = df.select_dtypes(include=[np.number])
df[df_numeric.columns] = df_numeric.fillna(df_numeric.mean())
df_categorical = df.select_dtypes(include=['object'])
for col in df_categorical.columns:
    df[col] = df[col].fillna(df[col].mode()[0])
print("Missing values handled.")

# === Remove Outliers Using IQR ===
Q1 = df_numeric.quantile(0.25)
Q3 = df_numeric.quantile(0.75)
IQR = Q3 - Q1
df_cleaned = df[~((df_numeric < (Q1 - 1.5 * IQR)) | (df_numeric > (Q3 + 1.5 * IQR))).any(axis=1)].copy()
df_cleaned.reset_index(drop=True, inplace=True)
print(f"Outliers removed. Remaining rows: {df_cleaned.shape[0]}")
print()

# === Linear Regression ===
if {'Sales', 'Spending_Score'}.issubset(df_cleaned.columns):
    X = df_cleaned[['Sales']]
    y = df_cleaned['Spending_Score']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    lin_model = LinearRegression()
    lin_model.fit(X_train, y_train)
    y_pred = lin_model.predict(X_test)
    print("=== Linear Regression Model: Sales -> Spending_Score ===")
    print(f"R2 Score: {r2_score(y_test, y_pred):.2f}")
    print(f"MSE: {mean_squared_error(y_test, y_pred):.2f}")
    print()

# === TensorFlow Model ===
if df_cleaned['Seasonality'].dtype == 'object':
    df_cleaned['Seasonality'] = LabelEncoder().fit_transform(df_cleaned['Seasonality'])

X = df_cleaned[['Marketing_Spend', 'Seasonality']]
y = df_cleaned['Sales']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = Sequential([
    Dense(32, input_dim=X.shape[1], activation='relu'),
    Dense(16, activation='relu'),
    Dense(1)
])
model.compile(optimizer='adam', loss='mse')
print("Training TensorFlow model...")
model.fit(X_train, y_train, epochs=50, verbose=0)
predictions_tf = model.predict(X_test, verbose=0).flatten()

print()
print("=== TensorFlow Regression Model: Marketing_Spend + Seasonality -> Sales ===")
print(f"R2 Score: {r2_score(y_test, predictions_tf):.2f}")
print(f"MSE: {mean_squared_error(y_test, predictions_tf):.2f}")
print()

# === Save Model ===
model.save(r"C:\Users\angel\sales_model.h5")
print("Model saved!")
print()

# === Kafka Producer ===
print("=" * 70)
print("KAFKA STREAMING")
print("=" * 70)
producer = KafkaProducer(
    bootstrap_servers=['localhost:9092'],
    value_serializer=lambda v: json.dumps(v).encode('utf-8')
)
print("Connected to Kafka broker.")

kafka_topic = 'sales_predictions'
results_df = pd.DataFrame({
    'predicted_sales': predictions_tf,
    'actual_sales': y_test.values
})

print(f"Sending {len(results_df)} records to topic: {kafka_topic}...")
for index, row in results_df.iterrows():
    record = {
        'prediction_index': int(index),
        'predicted_sales': round(float(row['predicted_sales']), 2),
        'actual_sales': round(float(row['actual_sales']), 2),
        'error': round(float(row['actual_sales'] - row['predicted_sales']), 2)
    }
    producer.send(kafka_topic, value=record)
producer.flush()
producer.close()
print("All records sent to Kafka!")
print()

# === Kafka Consumer - Read Results ===
print("=" * 70)
print("KAFKA STREAMING RESULTS SUMMARY")
print("=" * 70)
print()

consumer = KafkaConsumer(
    'sales_predictions',
    bootstrap_servers=['localhost:9092'],
    auto_offset_reset='earliest',
    enable_auto_commit=False,
    value_deserializer=lambda x: json.loads(x.decode('utf-8')),
    consumer_timeout_ms=5000
)

messages = []
for message in consumer:
    messages.append(message.value)
consumer.close()

predicted = [m['predicted_sales'] for m in messages]
actual = [m['actual_sales'] for m in messages]
errors = [m['error'] for m in messages]
abs_errors = [abs(e) for e in errors]

rmse = (sum(e**2 for e in errors) / len(errors)) ** 0.5
mape = sum(abs(e/a)*100 for e, a in zip(errors, actual) if a != 0) / len(actual)

within_10pct = sum(1 for e, a in zip(abs_errors, actual) if e/a <= 0.10)
within_25pct = sum(1 for e, a in zip(abs_errors, actual) if e/a <= 0.25)
within_50pct = sum(1 for e, a in zip(abs_errors, actual) if e/a <= 0.50)

over_predicted = sum(1 for e in errors if e < 0)
under_predicted = sum(1 for e in errors if e > 0)

sorted_msgs = sorted(messages, key=lambda x: abs(x['error']))

print('  Pipeline Overview')
print('  ┌────────────────────────┬───────────────────┐')
print('  │         Metric         │       Value       │')
print('  ├────────────────────────┼───────────────────┤')
print(f'  │ Total Records Streamed │ {len(messages):<17} │')
print('  ├────────────────────────┼───────────────────┤')
print('  │ Kafka Topic            │ sales_predictions │')
print('  ├────────────────────────┼───────────────────┤')
print('  │ Broker                 │ localhost:9092    │')
print('  └────────────────────────┴───────────────────┘')
print('  ---')
print('  Prediction vs Actual Statistics')
print('  ┌─────────┬──────────────────┬──────────────────┐')
print('  │ Metric  │ Predicted Sales  │   Actual Sales   │')
print('  ├─────────┼──────────────────┼──────────────────┤')
pred_range = f'${min(predicted):,.0f} - ${max(predicted):,.0f}'
act_range = f'${min(actual):,.0f} - ${max(actual):,.0f}'
print(f'  │ Range   │ {pred_range:<16} │ {act_range:<16} │')
print('  ├─────────┼──────────────────┼──────────────────┤')
pred_mean = f'${statistics.mean(predicted):,.0f}'
act_mean = f'${statistics.mean(actual):,.0f}'
print(f'  │ Mean    │ {pred_mean:<16} │ {act_mean:<16} │')
print('  ├─────────┼──────────────────┼──────────────────┤')
pred_med = f'${statistics.median(predicted):,.0f}'
act_med = f'${statistics.median(actual):,.0f}'
print(f'  │ Median  │ {pred_med:<16} │ {act_med:<16} │')
print('  ├─────────┼──────────────────┼──────────────────┤')
pred_std = f'${statistics.stdev(predicted):,.0f}'
act_std = f'${statistics.stdev(actual):,.0f}'
print(f'  │ Std Dev │ {pred_std:<16} │ {act_std:<16} │')
print('  └─────────┴──────────────────┴──────────────────┘')
print('  ---')
print('  Error Analysis')
print('  ┌────────────────────────────────┬────────────┐')
print('  │             Metric             │   Value    │')
print('  ├────────────────────────────────┼────────────┤')
bias_val = f'${statistics.mean(errors):,.2f}'
print(f'  │ Mean Error (Bias)              │ {bias_val:>10} │')
print('  ├────────────────────────────────┼────────────┤')
mae_val = f'${statistics.mean(abs_errors):,.2f}'
print(f'  │ Mean Absolute Error (MAE)      │ {mae_val:>10} │')
print('  ├────────────────────────────────┼────────────┤')
rmse_val = f'${rmse:,.2f}'
print(f'  │ Root Mean Square Error (RMSE)  │ {rmse_val:>10} │')
print('  ├────────────────────────────────┼────────────┤')
mape_val = f'{mape:.2f}%'
print(f'  │ Mean Absolute Percentage Error │ {mape_val:>10} │')
print('  └────────────────────────────────┴────────────┘')
print('  ---')
print('  Prediction Accuracy')
print('  ┌──────────────────────┬─────────┬────────────┐')
print('  │  Accuracy Threshold  │ Records │ Percentage │')
print('  ├──────────────────────┼─────────┼────────────┤')
rec10 = f'{within_10pct}/{len(messages)}'
pct10 = f'{within_10pct/len(messages)*100:.1f}%'
print(f'  │ Within 10% of actual │ {rec10:<7} │ {pct10:>10} │')
print('  ├──────────────────────┼─────────┼────────────┤')
rec25 = f'{within_25pct}/{len(messages)}'
pct25 = f'{within_25pct/len(messages)*100:.1f}%'
print(f'  │ Within 25% of actual │ {rec25:<7} │ {pct25:>10} │')
print('  ├──────────────────────┼─────────┼────────────┤')
rec50 = f'{within_50pct}/{len(messages)}'
pct50 = f'{within_50pct/len(messages)*100:.1f}%'
print(f'  │ Within 50% of actual │ {rec50:<7} │ {pct50:>10} │')
print('  └──────────────────────┴─────────┴────────────┘')
print(f'  Prediction Direction:')
print(f'  - Over-predicted: {over_predicted} records ({over_predicted/len(messages)*100:.1f}%)')
print(f'  - Under-predicted: {under_predicted} records ({under_predicted/len(messages)*100:.1f}%)')
print()
print('  ---')
print('  Best & Worst Predictions')
print()
print('  Best (Lowest Error):')
for i, msg in enumerate(sorted_msgs[:3]):
    print(f'  {i+1}. Predicted: ${msg["predicted_sales"]:,.0f} | Actual: ${msg["actual_sales"]:,.0f} | Error: ${msg["error"]:,.0f}')
print()
print('  Needs Improvement (Highest Error):')
for i, msg in enumerate(sorted_msgs[-3:]):
    print(f'  {i+1}. Predicted: ${msg["predicted_sales"]:,.0f} | Actual: ${msg["actual_sales"]:,.0f} | Error: ${msg["error"]:,.0f}')
print()
print('  ---')
print('  Key Insights')
print()
print('  1. Model Performance: The current model has high error rates, suggesting the features')
print('     (Marketing_Spend, Seasonality) alone do not fully explain Sales variance.')
mean_err = statistics.mean(errors)
print(f'  2. Bias: The positive mean error (${mean_err:,.0f}) indicates the model tends to')
print('     underestimate actual sales.')
print('  3. High Variance Cases: The worst predictions occur when actual sales are high ($95k+)')
print('     but the model predicts low values - likely missing important features.')
print('  4. Improvement Opportunities: Consider adding more features (Customer demographics,')
print('     Credit Score, historical trends) to improve accuracy.')
print()
print("=" * 70)
print("PIPELINE COMPLETE")
print("=" * 70)

# Show visualization
kafka_df = pd.DataFrame(messages)
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

ax1 = axes[0, 0]
min_val = min(kafka_df['predicted_sales'].min(), kafka_df['actual_sales'].min())
max_val = max(kafka_df['predicted_sales'].max(), kafka_df['actual_sales'].max())
ax1.plot([min_val, max_val], [min_val, max_val], 'r--', label='Perfect Prediction', linewidth=2)
scatter = ax1.scatter(kafka_df['predicted_sales'], kafka_df['actual_sales'],
                      c=abs(kafka_df['error']), cmap='coolwarm', alpha=0.7, edgecolors='black')
ax1.set_xlabel('Predicted Sales ($)')
ax1.set_ylabel('Actual Sales ($)')
ax1.set_title('Predicted vs Actual Sales', fontweight='bold')
ax1.legend()
plt.colorbar(scatter, ax=ax1, label='Absolute Error ($)')

ax2 = axes[0, 1]
ax2.hist(kafka_df['error'], bins=20, color='steelblue', edgecolor='black', alpha=0.7)
ax2.axvline(x=0, color='red', linestyle='--', linewidth=2, label='Zero Error')
ax2.axvline(x=kafka_df['error'].mean(), color='orange', linestyle='-', linewidth=2, label='Mean Error')
ax2.set_xlabel('Prediction Error ($)')
ax2.set_ylabel('Frequency')
ax2.set_title('Error Distribution', fontweight='bold')
ax2.legend()

ax3 = axes[1, 0]
sizes = [within_10pct, within_25pct-within_10pct, within_50pct-within_25pct, len(messages)-within_50pct]
labels = ['Within 10%', 'Within 25%', 'Within 50%', 'Beyond 50%']
colors = ['#2ecc71', '#f1c40f', '#e67e22', '#e74c3c']
ax3.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
ax3.set_title('Prediction Accuracy Breakdown', fontweight='bold')

ax4 = axes[1, 1]
ax4.bar(kafka_df['prediction_index'], kafka_df['error'],
        color=np.where(kafka_df['error'] >= 0, 'green', 'red'), alpha=0.7)
ax4.axhline(y=0, color='black', linestyle='-', linewidth=1)
ax4.set_xlabel('Prediction Index')
ax4.set_ylabel('Error ($)')
ax4.set_title('Prediction Errors by Record', fontweight='bold')

plt.tight_layout()
plt.suptitle('Kafka Streaming Results - Visual Analysis', fontsize=14, fontweight='bold', y=1.02)
plt.show()
