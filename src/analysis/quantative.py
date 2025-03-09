import pandas as pd


def get_feedback_from_user(user_inputs: list[list[dict]], num_questions: int = 3) -> pd.DataFrame:
    user_feedback = []

    for ii, user_input in enumerate(user_inputs):
        user_input = user_input["messages"]
        question_scores: list[float | None] = [None for _ in range(num_questions)]

        survey_index = None
        for item_id, item in enumerate(user_input):
            if item_id != 0 and "survey".lower() in item["content"]:
                survey_index = item_id
                break
        else:
            print(f"No feedback shared by user: {ii}")

        if survey_index is not None:
            try:
                question_scores[0] = user_input[survey_index + 3]["content"]
                question_scores[1] = user_input[survey_index + 5]["content"]
                question_scores[2] = user_input[survey_index + 7]["content"]
            except IndexError:
                pass
        user_feedback.append(question_scores)

    feedback_df = pd.DataFrame(user_feedback, columns=["Q1", "Q2", "Q3"])

    return feedback_df


def get_quantative_analysis(dataset: pd.DataFrame) -> pd.DataFrame:
    failed_conversations = [True] + [False for _ in range(len(dataset) - 1)]

    feedback_df = get_feedback_from_user(dataset["inputs"], num_questions=3)

    conversation_length = [len(user_input["messages"]) for user_input in dataset["inputs"]]

    number_of_user_words = []
    for user_input in dataset["inputs"].tolist():
        user_input = user_input["messages"]
        user_input = [item["content"] for item in user_input if item["role"] == "user"]
        user_words = [item.split() for item in user_input]
        number_of_user_words.append(sum(len(word) for word in user_words))

    output_df = pd.DataFrame(
        {
            "failed_conversation": failed_conversations,
            "conversation_length": conversation_length,
            "number_of_user_words": number_of_user_words,
            "feedback_Q1": feedback_df["Q1"],
            "feedback_Q2": feedback_df["Q2"],
            "feedback_Q3": feedback_df["Q3"],
        }
    )

    return output_df
