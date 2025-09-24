import pandas as pd
import numpy as np
from sklearn.metrics import classification_report

CPU_THRESHOLD = 0.5

ALERT_WINDOW = 2

def run_reactive_baseline(data_path='processed_temporal_data.csv'):
    """
    Simulates a traditional, threshold-based reactive monitoring system.
    """
    print("Running reactive baseline simulation...")

    df = pd.read_csv(data_path)

    MICROSERVICES = [
        'frontend', 'cartservice', 'productcatalogservice', 'currencyservice',
        'paymentservice', 'shippingservice', 'emailservice', 'checkoutservice',
        'recommendationservice', 'adservice'
    ]

    label_cols = [f'{s}_label' for s in MICROSERVICES]
    cpu_cols = [f'{s}_cpu_usage' for s in MICROSERVICES]

    true_labels = df[label_cols]
    cpu_data = df[cpu_cols]

    from sklearn.preprocessing import MinMaxScaler
    scaler = MinMaxScaler()
    scaled_cpu_data = scaler.fit_transform(cpu_data)
    scaled_cpu_df = pd.DataFrame(scaled_cpu_data, columns=cpu_cols)

    predictions = pd.DataFrame(0, index=df.index, columns=label_cols)

    print(f"Applying alert rule: CPU > {CPU_THRESHOLD} for {ALERT_WINDOW} consecutive snapshots.")

    for service_cpu_col, service_label_col in zip(cpu_cols, label_cols):
        is_breached = scaled_cpu_df[service_cpu_col].rolling(window=ALERT_WINDOW).min() > CPU_THRESHOLD

        predictions.loc[is_breached, service_label_col] = 1

    true_labels_flat = true_labels.values.flatten()
    predictions_flat = predictions.values.flatten()

    print("\n--- Reactive Baseline Evaluation Report ---")
    report = classification_report(
        true_labels_flat,
        predictions_flat,
        target_names=['Healthy', 'Faulty'],
        zero_division=0
    )
    print(report)
    return report

if __name__ == '__main__':
    run_reactive_baseline()
