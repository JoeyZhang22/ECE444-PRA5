import requests

# Define your API endpoint
API_URL = "http://server-sentiment-env.eba-umns7bbr.us-east-2.elasticbeanstalk.com/test_prediction"

# Define test cases and expected outcomes
test_cases = [
    {"input": "The earth is flat", "expected": "real"},
    {"input": "The stock market is closed on weekends", "expected": "real"},
    {"input": "Aliens built the pyramids", "expected": "real"},
    {"input": "Water boils at 100 degrees Celsius", "expected": "real"},
]

# Run each test case
for i, test_case in enumerate(test_cases, start=1):
    # Send the POST request with the test case input
    response = requests.post(API_URL, json={"text": test_case["input"]})

    result = response.json()

    print(result)
    prediction = result.get("prediction")  # Get "prediction" value ("fake" or "real")

    # Assert that the prediction matches the expected outcome
    assert (
        prediction == test_case["expected"]
    ), f"Test case {i} failed: expected '{test_case['expected']}' but got '{prediction}'"

    print(f"Test case {i} passed: '{test_case['input']}' -> Prediction: {prediction}")
