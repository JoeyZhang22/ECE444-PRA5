import requests
import time
import pandas as pd
import matplotlib.pyplot as plt

API_URL = "http://127.0.0.1:5002/test_prediction"

test_cases = [  # All test cases are fake but the models on my end always return REAL
    {"input": "Fake news"},
    {"input": "I am Joey"},
    {"input": "I am not Joey"},
    {"input": "Earth is flat"},
]

# List to store latency data
latency_data = []

# Perform latency testing
for test_case in test_cases:
    for _ in range(100):  # 100 API calls per test case
        start_time = time.time()

        # Send the request and measure latency
        response = requests.post(API_URL, json={"input": test_case["input"]})
        latency = time.time() - start_time

        # Append to latency_data with seconds only
        latency_data.append(
            {
                "timestamp": int(time.time() % 60),  # Only seconds
                "input": test_case["input"],
                "latency": latency,
            }
        )

# Calculate and print average latency
total_latency = sum(item["latency"] for item in latency_data)
average_latency = total_latency / len(latency_data)
print(f"Average latency: {average_latency:.4f} seconds")

# Create a DataFrame from the collected data
df = pd.DataFrame(latency_data)

# Save to CSV file
df.to_csv("latency_results.csv", index=False)

# Generate a boxplot for latency with adjusted spacing
plt.figure(figsize=(14, 6))
plt.boxplot(
    [df[df["input"] == case]["latency"] for case in df["input"].unique()],
    labels=df["input"].unique(),
    widths=0.5,
)
plt.title("Latency for Different Test Cases")
plt.ylabel("Latency (seconds)")
plt.xlabel("Input Text")
plt.xticks(rotation=15)
plt.grid()
plt.savefig("latency_boxplot.png")
plt.show()
