import pandas as pd


def calculate_overall_feedback(row):
    # If all feedback columns are empty/null, return None
    if pd.isna(row["feedback_Q1"]) and pd.isna(row["feedback_Q2"]) and pd.isna(row["feedback_Q3"]):
        return None

    # Convert feedback to numerical scores
    scores = []

    for feedback_key in ["feedback_Q1", "feedback_Q2", "feedback_Q3"]:
        if not pd.isna(row[feedback_key]):
            feedback = str(row[feedback_key]).lower()

            if any(phrase in feedback for phrase in ["extremely helpful", "very satisfactory"]):
                scores.append(5)
            elif any(phrase in feedback for phrase in ["helped a lot", "very helpful"]):
                scores.append(4)
            elif any(phrase in feedback for phrase in ["satisfactory", "it helped"]):
                scores.append(3)
            elif any(phrase in feedback for phrase in ["somewhat helpful", "slightly"]):
                scores.append(2)
            elif "not" in feedback or "negative" in feedback:
                scores.append(1)
            else:
                # Default to 3 if we can't determine the sentiment
                scores.append(3)

    # return round(sum(scores) / len(scores))
    return max(scores)


def get_feedback_from_user(user_inputs: list[list[dict]], num_questions: int = 3) -> pd.DataFrame:
    """In conversation length and user words we do not include feedback messages."""
    user_feedback = []
    conversation_length = []
    user_words = []

    for ii, user_input in enumerate(user_inputs):
        user_input = user_input["messages"]
        question_scores: list[float | None] = [None for _ in range(num_questions)]

        survey_index = None
        for item_id, item in enumerate(user_input):
            if item_id != 0 and "how would you rate".lower() in item["content"].lower():
                survey_index = item_id
                break
        else:
            print(f"No feedback shared by user: {ii}")

        conversation_length.append(survey_index if survey_index is not None else len(user_input))
        user_words.append(
            sum(len(item["content"].split()) for item in user_input[:survey_index] if item["role"] == "user")
        )

        if survey_index is not None:
            try:
                question_scores[0] = user_input[survey_index + 1]["content"]
                question_scores[1] = user_input[survey_index + 3]["content"]
                question_scores[2] = user_input[survey_index + 5]["content"]
            except IndexError:
                pass
        user_feedback.append(question_scores)

    feedback_df = pd.DataFrame(user_feedback, columns=["Q1", "Q2", "Q3"])
    feedback_df["conversation_length"] = conversation_length
    feedback_df["user_words"] = user_words
    return feedback_df


def get_quantative_analysis(dataset: pd.DataFrame) -> pd.DataFrame:
    feedback_df = get_feedback_from_user(dataset["inputs"], num_questions=3)

    output_df = pd.DataFrame(
        {
            "conversation_id": list(range(len(dataset))),
            "conversation_length": feedback_df["conversation_length"],
            "user_words": feedback_df["user_words"],
            "api_errors": dataset["api_errors"],
            "feedback_Q1": feedback_df["Q1"],
            "feedback_Q2": feedback_df["Q2"],
            "feedback_Q3": feedback_df["Q3"],
        }
    )

    output_df["overall_feedback"] = output_df.apply(calculate_overall_feedback, axis=1)

    return output_df
