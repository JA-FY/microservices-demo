import pandas as pd
import numpy as np

TRAIN_SPLIT_RATIO = 0.03
VAL_SPLIT_RATIO = 0.06
CPU_THRESHOLD = 0.5
ALERT_WINDOW = 2
SECONDS_PER_SNAPSHOT = 30

def analyze_detection_time():
    print("Analyzing detection times for TGN vs. Reactive Baseline...")

    df = pd.read_csv('processed_temporal_data.csv')

    num_snapshots = len(df)
    train_snap_idx = int(num_snapshots * TRAIN_SPLIT_RATIO)
    val_snap_idx = int(num_snapshots * VAL_SPLIT_RATIO)

    validation_df = df.iloc[train_snap_idx:val_snap_idx].reset_index(drop=True)
    print(f"Isolating validation set (snapshots {train_snap_idx} to {val_snap_idx})...")

    MICROSERVICES = ['frontend', 'cartservice', 'productcatalogservice', 'currencyservice','paymentservice', 'shippingservice', 'emailservice', 'checkoutservice','recommendationservice', 'adservice']
    label_cols = [f'{s}_label' for s in MICROSERVICES]
    cpu_cols = [f'{s}_cpu_usage' for s in MICROSERVICES]

    from sklearn.preprocessing import MinMaxScaler
    scaler = MinMaxScaler()
    scaled_cpu_data = scaler.fit_transform(validation_df[cpu_cols])
    scaled_cpu_df = pd.DataFrame(scaled_cpu_data, columns=cpu_cols)

    reactive_predictions_df = pd.DataFrame(0, index=validation_df.index, columns=label_cols)
    for cpu_col, label_col in zip(cpu_cols, label_cols):
        is_breached = scaled_cpu_df[cpu_col].rolling(window=ALERT_WINDOW).min() > CPU_THRESHOLD
        reactive_predictions_df.loc[is_breached, label_col] = 1

    try:
        tgn_results = pd.read_csv('validation_predictions.csv')
    except FileNotFoundError:
        print("\nERROR: `validation_predictions.csv` not found.")
        print("Please re-run `final_run.py` to generate the prediction file.")
        return

    true_labels_flat = tgn_results['true_labels'].values
    tgn_predictions_flat = tgn_results['predictions'].values
    reactive_preds_flat = reactive_predictions_df.values.flatten()

    fault_indices = np.where(true_labels_flat == 1)[0]

    if len(fault_indices) == 0:
        print("No true faults found in the validation set to analyze.")
        return

    first_true_fault_idx_relative = fault_indices[0]

    reactive_alarm_indices = np.where((reactive_preds_flat == 1) & (true_labels_flat == 1))[0]
    t_breach_relative = reactive_alarm_indices[0] if len(reactive_alarm_indices) > 0 else float('inf')

    tgn_alarm_indices = np.where((tgn_predictions_flat == 1) & (true_labels_flat == 1))[0]
    t_prediction_relative = tgn_alarm_indices[0] if len(tgn_alarm_indices) > 0 else float('inf')

    print("\n--- Results (within the validation set) ---")
    print(f"First actual fault occurs at relative index: {first_true_fault_idx_relative}")
    print(f"Reactive baseline first detected a fault at relative index: {'N/A' if t_breach_relative == float('inf') else t_breach_relative}")
    print(f"TGN model first detected a fault at relative index: {'N/A' if t_prediction_relative == float('inf') else t_prediction_relative}")

    if t_breach_relative != float('inf'):
        punctuality_snapshots = first_true_fault_idx_relative - t_breach_relative
        punctuality_seconds = punctuality_snapshots * SECONDS_PER_SNAPSHOT
        print(f"\nReactive Baseline 'Predictive Punctuality': {punctuality_seconds:.2f} seconds.")
        print("(A negative value means it detected the fault AFTER it occurred, which is expected).")

    if t_prediction_relative != float('inf'):
         punctuality_snapshots = first_true_fault_idx_relative - t_prediction_relative
         punctuality_seconds = punctuality_snapshots * SECONDS_PER_SNAPSHOT
         print(f"TGN 'Predictive Punctuality': {punctuality_seconds:.2f} seconds.")
    else:
        print("TGN did not detect any faults, so Predictive Punctuality cannot be calculated.")


if __name__ == '__main__':
    analyze_detection_time()
