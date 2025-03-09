import pandas as pd


def remove_duplicated_messages(messages: list[dict]) -> list[dict]:
    cleaned_messages = []

    prev_message = None
    for message in messages:
        if prev_message == message["content"]:
            continue
        prev_message = message["content"]
        cleaned_messages.append(message)

    return cleaned_messages


def remove_assistant_api_error_messages(messages: list[dict]) -> list[dict]:
    return [
        message
        for message in messages
        if "Incorrect API key provided" not in message["content"]
        and "You exceeded your current quota" not in message["content"]
    ]


def load_dataset(dataset_path: str) -> pd.DataFrame:
    dataset = pd.read_json(dataset_path, orient="records", lines=True)
    user_inputs = dataset["inputs"]

    preprocessed_user_inputs = []
    for user_input in user_inputs:
        user_input = user_input["messages"]

        user_input = remove_assistant_api_error_messages(user_input)
        user_input = remove_duplicated_messages(user_input)
        preprocessed_user_inputs.append({"messages": user_input})

    dataset["inputs"] = preprocessed_user_inputs
    return dataset
