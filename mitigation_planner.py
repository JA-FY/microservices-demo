from kubernetes import client, config
import time

# KNOWLEDGE (K) of the MAPE-K Loop
SERVICE_DEPENDENCIES = {
    'productcatalogservice': ['frontend', 'checkoutservice', 'cartservice', 'recommendationservice'],
    'currencyservice': ['frontend', 'checkoutservice', 'shippingservice'],
    'shippingservice': ['frontend', 'checkoutservice'],
    'cartservice': ['frontend', 'checkoutservice'],
}

# PLAN (P) of the MAPE-K Loop
def get_mitigation_plan(predicted_faulty_service: str) -> list:
    """
    Generates a topology-conscious mitigation plan based on a prediction.
    """
    print(f"PLAN: A fault is predicted for '{predicted_faulty_service}'. Generating mitigation plan.")

    plans = []

    plan_A = {
        "action": "scale_up",
        "target_service": predicted_faulty_service,
        "replicas": 2,
        "reason": f"Reinforce the predicted failing service '{predicted_faulty_service}'."
    }
    plans.append(plan_A)

    if predicted_faulty_service in SERVICE_DEPENDENCIES:
        dependent_services = SERVICE_DEPENDENCIES[predicted_faulty_service]
        for service in dependent_services:
            plan_B = {
                "action": "scale_up",
                "target_service": service,
                "replicas": 2,
                "reason": f"Proactively reinforce '{service}' as it depends on the predicted failing service."
            }
            plans.append(plan_B)

    return plans

# EXECUTE (E) of the MAPE-K Loop
def execute_plan(plan: dict):
    """
    Connects to the Kubernetes API to execute a given plan.
    """
    print(f"EXECUTE: Executing plan -> {plan['reason']}")

    try:
        config.load_kube_config()
        api = client.AppsV1Api()

        deployment_name = plan['target_service']

        if plan['action'] == 'scale_up':
            body = {"spec": {"replicas": plan['replicas']}}
            api.patch_namespaced_deployment_scale(name=deployment_name, namespace=namespace, body=body)
            print(f"  - SUCCESS: Scaled deployment '{deployment_name}' to {plan['replicas']} replicas.")
        else:
            print(f"  - WARNING: Action '{plan['action']}' is not implemented.")

    except client.ApiException as e:
        if e.status == 404:
            print(f"  - ERROR: Deployment '{plan['target_service']}' not found in namespace '{namespace}'.")
        else:
            print(f"  - ERROR: Kubernetes API error. {e.reason}")
    except Exception as e:
        print(f"  - ERROR: Could not execute plan. Is your kubeconfig set up correctly? {e}")

if __name__ == '__main__':
    print("--- Proactive Mitigation System Simulation ---")

    # Monitor (M)
    print("\nMONITOR: (Simulating a stream of observability data...)\n")
    time.sleep(1)


    # Analyze (A)
    hypothetical_prediction = "productcatalogservice"
    print(f"ANALYZE: TGN model predicts imminent failure for '{hypothetical_prediction}' with high confidence.\n")
    time.sleep(1)

    # Plan (P)
    list_of_plans = get_mitigation_plan(hypothetical_prediction)
    print(f"\n  - Plan generation complete. {len(list_of_plans)} actions to execute.\n")
    time.sleep(1)

    # Execute (E)
    for p in list_of_plans:
        execute_plan(p)
        time.sleep(0.5)

    print("\n--- MAPE-K Loop Simulation Complete ---")
