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


def remove_assistant_api_error_messages(messages: list[dict]) -> tuple[list[dict], int]:
    error_count = 0
    cleaned_messages = []

    for i, message in enumerate(messages):
        if (
            "Incorrect API key provided" in message["content"]
            or "You exceeded your current quota" in message["content"]
        ):
            error_count += 1
            # Skip this message and the previous one if it exists
            if i > 0 and cleaned_messages:
                cleaned_messages.pop()
        else:
            cleaned_messages.append(message)

    return cleaned_messages, error_count


def load_dataset(dataset_path: str) -> pd.DataFrame:
    dataset = pd.read_json(dataset_path, orient="records", lines=True)
    user_inputs = dataset["inputs"]

    preprocessed_user_inputs = []
    api_errors = []
    for user_input in user_inputs:
        user_input = user_input["messages"]

        user_input, error_count = remove_assistant_api_error_messages(user_input)
        user_input = remove_duplicated_messages(user_input)
        preprocessed_user_inputs.append({"messages": user_input})
        api_errors.append(error_count)

    dataset["inputs"] = preprocessed_user_inputs
    dataset["api_errors"] = api_errors
    dataset["conversation_id"] = dataset.index
    return dataset
