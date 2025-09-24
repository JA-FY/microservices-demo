import json
import pandas as pd
from datetime import datetime, timezone
from prometheus_api_client import PrometheusConnect
import warnings

warnings.filterwarnings('ignore', category=FutureWarning)

def fetch_and_label_data(prometheus_url, chaos_log_path, services, exp_to_service_map):
    prom = PrometheusConnect(url=prometheus_url, disable_ssl=True)

    with open(chaos_log_path, 'r') as f:
        chaos_logs = json.load(f)

#    min_start = min(datetime.fromisoformat(log['start_time_utc']).timestamp() for log in chaos_logs)
#    max_end = max(datetime.fromisoformat(log['end_time_utc']).timestamp() for log in chaos_logs)
#    start_time = datetime.fromtimestamp(min_start, tz=timezone.utc)
#    end_time = datetime.fromtimestamp(max_end, tz=timezone.utc)

    start_time = datetime(2025, 9, 23, 22, 0, 0, tzinfo=timezone.utc)
    end_time = datetime(2025, 9, 24, 6, 0, 0, tzinfo=timezone.utc)
    print(f"Querying data from {start_time} to {end_time}")

    time_index = pd.date_range(start=start_time, end=end_time, freq='30s')
    all_metrics_df = pd.DataFrame(index=time_index)

    metric_queries = {
        'cpu_usage': 'rate(container_cpu_usage_seconds_total{{namespace="default", pod=~"^({service_name})-.*"}}[2m])',
        'memory_usage': 'container_memory_working_set_bytes{{namespace="default", pod=~"^({service_name})-.*"}}',
    }

    for service in services:
        print(f"Fetching metrics for service: {service}")
        for metric_name, query_template in metric_queries.items():
            query = query_template.format(service_name=service)
            try:
                metric_data = prom.custom_query_range(
                    query=query, start_time=start_time, end_time=end_time, step='30s'
                )
                if metric_data:
                    res = metric_data[0]
                    col_name = f'{service}_{metric_name}'
                    df = pd.DataFrame(res['values'], columns=['timestamp', col_name])

                    df[col_name] = pd.to_numeric(df[col_name])

                    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s', utc=True)
                    df.set_index('timestamp', inplace=True)
                    all_metrics_df = all_metrics_df.join(df, how='left')
                else:
                    print(f"  - No data for {service} on metric {metric_name}")
            except Exception as e:
                print(f"  - Could not fetch {metric_name} for {service}. Error: {e}")

    all_metrics_df.interpolate(method='time', inplace=True)
    all_metrics_df.ffill(inplace=True)
    all_metrics_df.bfill(inplace=True)

    print("\n--- Generating Ground Truth Labels ---")
    for service in services:
        all_metrics_df[f'{service}_label'] = 0

    for log in chaos_logs:
        exp_id = log['experiment_id']
        service_name = exp_to_service_map.get(exp_id)
        if not service_name:
            print(f"  - ⚠️ Warning: No service mapping for experiment '{exp_id}'. Skipping labeling.")
            continue
        start = datetime.fromisoformat(log['start_time_utc'])
        end = datetime.fromisoformat(log['end_time_utc'])
        all_metrics_df.loc[
            (all_metrics_df.index >= start) & (all_metrics_df.index <= end),
            f'{service_name}_label'
        ] = 1
        print(f"  - Labeled '{service_name}' as faulty for '{exp_id}'")

    all_metrics_df.reset_index(inplace=True)
    all_metrics_df.rename(columns={'index': 'timestamp_utc'}, inplace=True)
    all_metrics_df['unix_timestamp'] = all_metrics_df['timestamp_utc'].apply(lambda x: x.timestamp())
    return all_metrics_df

if __name__ == "__main__":
    MICROSERVICES = [
        'frontend', 'cartservice', 'productcatalogservice', 'currencyservice',
        'paymentservice', 'shippingservice', 'emailservice', 'checkoutservice',
        'recommendationservice', 'adservice'
    ]

    EXPERIMENT_TO_SERVICE = {
        "T1_DNS_FAILURE_SUSTAINED": "checkoutservice",
        "T2_THROTTLED_DEPENDENCY_RUN_1": "recommendationservice",
        "T3_FAULTY_ROLLING_UPDATE_RUN_1": "frontend",
        "T2.5_WIDE_AREA_FLAPPING_CHAOS": "cartservice",
        "T1_POD_OUTAGE_SCHEDULED": "recommendationservice"
    }

    processed_data = fetch_and_label_data(
        prometheus_url='http://localhost:9090',
        chaos_log_path='./chaos_log.json',
        services=MICROSERVICES,
        exp_to_service_map=EXPERIMENT_TO_SERVICE
    )

    print("\n--- Final Processed DataFrame (first 5 rows) ---")
    print(processed_data.head())
    processed_data.to_csv('processed_temporal_data.csv', index=False)
