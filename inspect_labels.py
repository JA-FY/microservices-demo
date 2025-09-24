import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('processed_temporal_data.csv')
label_cols = [col for col in df.columns if '_label' in col]
df['any_fault_active'] = df[label_cols].max(axis=1)

faulty_snapshots = df[df['any_fault_active'] == 1]
if not faulty_snapshots.empty:
    last_fault_index = faulty_snapshots.index[-1]
    print(f"✅ Last fault occurs at snapshot index: {last_fault_index}")
else:
    print("🚨 No faults found in the dataset.")
print(f"Total number of time snapshots: {len(df)}")

plt.figure(figsize=(15, 5))
plt.plot(df.index, df['any_fault_active'], drawstyle='steps-post')
plt.title('Fault Activity Over Time (by Snapshot Index)')
plt.xlabel('Snapshot Index')
plt.ylabel('Fault Active (1=Yes, 0=No)')
plt.grid(True)
plt.show()
