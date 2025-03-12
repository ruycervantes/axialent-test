import streamlit as st
import pandas as pd
import plotly.graph_objects as go

from src.data_preparation import load_dataset
import plotly.express as px
from plotly.subplots import make_subplots


dataset_path = "data/dataset.jsonl"

dataset = load_dataset(dataset_path)

st.title("Axialent Test Challeng task")

st.markdown(
    "In this work I tried to analyse dataset with user communications with usage of LLM and langchain for finding patterns and insights."
)

st.subheader("Dataset description")

col1, col2 = st.columns(2)

with col1:
    sample = pd.DataFrame(dataset["inputs"][1]["messages"])
    st.dataframe(sample, hide_index=True, width=500)

with col2:
    st.write("Dataset contains 19 conversation between user and assistant.")
    st.write("First sample of dataset was removed due to api errors in assistant messages.")
    st.write("Dataset was preprocessed in following way:")
    st.markdown("""
    - Duplicate messages were removed
    - Assistant api error messages were removed
    - Text feedback was converted to numerical scores (from 1 to 5)
    - Conversation length and user words were calculated
    """)


## Qualitative Analysis

st.subheader("Qualitative Analysis")

col1, col2 = st.columns(spec=[0.6, 0.4])

with col1:
    quantative_df = pd.read_csv("data/quantative_df.csv")
    quantative_df = quantative_df[quantative_df["conversation_id"] != 0]

    # Filter out rows with empty feedback values
    filtered_df = quantative_df.dropna(subset=["overall_feedback"])

    feedback_values = filtered_df["overall_feedback"].unique()
    conversation_length_by_feedback = filtered_df.groupby("overall_feedback")["conversation_length"].mean()

    # Calculate mean for rows without feedback
    no_feedback_mean = quantative_df[quantative_df["overall_feedback"].isna()]["conversation_length"].mean()

    # Create a new Series with "No Feedback" as the first entry
    all_data = pd.Series([no_feedback_mean], index=["No Feedback"])
    all_data = pd.concat([all_data, conversation_length_by_feedback])

    all_data.index = all_data.index.astype(str)

    # Create bar colors (first gray, rest blue)
    colors = ["lightgray"] + ["skyblue"] * len(conversation_length_by_feedback)

    # Create the figure
    fig = go.Figure(
        data=[
            go.Bar(
                x=all_data.index,
                y=all_data.values,
                marker_color=colors,
                text=[f"{val:.1f}" for val in all_data.values],
                textposition="outside",
            )
        ]
    )

    # Update layout
    fig.update_layout(
        title="Average Conversation Length by Feedback Score",
        xaxis_title="Feedback Score",
        yaxis_title="Average Conversation Length",
        showlegend=False,
        template="plotly_white",
    )

    # Display the plotly figure in Streamlit
    st.plotly_chart(fig)

with col2:
    st.markdown(
        "Here I visualised average conversation length by feedback score (not including messages about feedback)."
    )
    st.markdown(
        "As we can see, there is no correlation between feedback score and conversation length - as number of questions is predefined in the prompt, conversation length stays the same."
    )


col1, col2 = st.columns(spec=[0.6, 0.4])


with col1:
    user_words_by_feedback = filtered_df.groupby("overall_feedback")["user_words"].mean()

    # Calculate mean for rows without feedback
    no_feedback_mean = quantative_df[quantative_df["overall_feedback"].isna()]["user_words"].mean()

    # Create a new Series with "No Feedback" as the first entry
    all_data = pd.Series([no_feedback_mean], index=["No Feedback"])
    all_data = pd.concat([all_data, user_words_by_feedback])

    all_data.index = all_data.index.astype(str)

    # Create bar colors (first gray, rest blue)
    colors = ["lightgray"] + ["skyblue"] * len(user_words_by_feedback)

    # Create the figure
    fig = go.Figure(
        data=[
            go.Bar(
                x=all_data.index,
                y=all_data.values,
                marker_color=colors,
                text=[f"{val:.1f}" for val in all_data.values],
                textposition="outside",
            )
        ]
    )

    # Update layout
    fig.update_layout(
        title="Average User Words by Feedback Score",
        xaxis_title="Feedback Score",
        yaxis_title="Average User Words",
        showlegend=False,
        template="plotly_white",
    )

    # Display the plotly figure in Streamlit
    st.plotly_chart(fig)


with col2:
    st.markdown(
        "Although we have only one feedback for 3.0, we can see that user rated 5.0 tends to user more words with same amount of messages."
    )
    st.markdown(
        "Thus, we should encourage users to use more words, explaining their difficult situations - possibly if user do not use more words, ask him more questions or directly ask him to explain his situation more detailed and deeper."
    )
    st.markdown("Hovewer more data is needed to make any conclusions.")


col1, col2 = st.columns(spec=[0.6, 0.4])

with col1:
    feedback_percentages = [
        len(quantative_df["feedback_Q1"].dropna()) / len(quantative_df) * 100,
        len(quantative_df["feedback_Q2"].dropna()) / len(quantative_df) * 100,
        len(quantative_df["feedback_Q3"].dropna()) / len(quantative_df) * 100,
    ]

    fig = go.Figure(
        data=[
            go.Bar(
                x=["Q1", "Q2", "Q3"],
                y=feedback_percentages,
                marker_color=["skyblue"] * len(feedback_percentages),
                text=[f"{val:.1f}" for val in feedback_percentages],
                textposition="outside",
            )
        ]
    )

    # Update layout
    fig.update_layout(
        title="Percentage of Feedback Responses by Question",
        xaxis_title="Question",
        yaxis_title="Percentage of Feedback Responses",
        showlegend=False,
        template="plotly_white",
    )

    # Display the plotly figure in Streamlit
    st.plotly_chart(fig)

with col2:
    st.markdown("Around 50% of user shared their feedback, that means users are happy to provide feedback.")
    st.markdown(
        "Also, percentage of feedback for each question is more or less the same, which means that users are consistent in providing feedback - no shift in feedback depending on the question."
    )

## User Communication Style Analysis and Prompts improvements
st.subheader("Analysis of main topic conversations")
st.markdown(
    "I used structured prompting to extract main topic and situation from user conversation: I listed all possible difficult topics and situations that can happen during work and asked model to select one of them (with temprature 0.0 for deterministic output)."
)

col1, col2 = st.columns(spec=[0.5, 0.5])
topics_df = pd.read_csv("data/topics_df.csv")

with col1:
    st.dataframe(topics_df[["main_topic", "situation"]], hide_index=True, height=350)


with col2:
    st.markdown("For main topic classes I used following list:")

    st.markdown("""
    -  Performance Related
    -  Interpersonal Conflicts
    -  Career Development
    -  Organizational Changes
    -  Professional Conduct
    -  Work Life Balance
    -  Ethics Compliance
    -  Project Management
    -  Client Relations
    -  Team Dynamics
    """)

st.markdown(
    "Although amount of situation is limited, but due to small amount of data, it is not possible to extract meaningful insights from it."
)
st.markdown("Later we analysed main topics in conversations matched with feedback score.")


col1, col2 = st.columns(spec=[0.7, 0.3])
topics_df = pd.read_csv("data/topics_df.csv")
topics_df = pd.merge(topics_df, quantative_df, on="conversation_id", how="left")

with col1:
    # Calculate mean feedback score for each main_topic and sort
    topic_feedback = topics_df.groupby("main_topic")["overall_feedback"].mean().sort_values(ascending=False)
    topic_feedback = topic_feedback.fillna(0)

    # Create a vertical bar chart with plotly
    fig = px.bar(
        y=topic_feedback.values,
        x=topic_feedback.index,
        labels={"y": "Average Feedback Score", "x": "Topic"},
        title="Average Feedback Score by Topic",
        color=topic_feedback.values,
        color_continuous_scale="Viridis",
    )

    # Update layout for better readability
    fig.update_layout(height=600, width=800, xaxis={"categoryorder": "total descending"}, coloraxis_showscale=False)

    st.plotly_chart(fig)


with col2:
    st.markdown(
        "As most of feedback we have seen is either 5.0 or no feedback, we can not tell for sure in which cases LLM works great and in which fails."
    )


# Create a figure with 2 subplots side by side
fig = make_subplots(
    rows=1,
    cols=3,
    subplot_titles=("All Topics", "Topics with Feedback", "Without Feedback"),
    specs=[[{"type": "pie"}, {"type": "pie"}, {"type": "pie"}]],
)

# First chart - all conversations
all_counts = topics_df["main_topic"].value_counts()
fig.add_trace(go.Pie(labels=all_counts.index, values=all_counts.values, hole=0.4, name="All Topics"), row=1, col=1)

# Second chart - only conversations with feedback
filtered_counts = topics_df.dropna(subset=["overall_feedback"])["main_topic"].value_counts()
fig.add_trace(
    go.Pie(labels=filtered_counts.index, values=filtered_counts.values, hole=0.4, name="With Feedback"), row=1, col=2
)

# Third chart - only conversations without feedback
filtered_counts = topics_df[topics_df["overall_feedback"].isna()]["main_topic"].value_counts()
fig.add_trace(
    go.Pie(labels=filtered_counts.index, values=filtered_counts.values, hole=0.4, name="Without Feedback"), row=1, col=3
)

# Update layout
fig.update_layout(title_text="Topic Distribution: All Conversations vs With Feedback")

st.plotly_chart(fig)
st.markdown(
    "As we can see, out of 4 conversation about Career Development, only one was given feedback (however by statics 50% of people left feedback). This is possible that LLM fails to analyse situactions about career and user stops in the middle of conversation."
)
st.markdown("For other topics we do not see any correlation between feedback and topic.")

st.subheader("Communication Style Analysis")

col1, col2 = st.columns(spec=[0.5, 0.5])

commucation_style_path = "data/communication_style.jsonl"
communication_style_df = pd.read_csv(commucation_style_path)
communication_style_df.loc[communication_style_df["emotional_tone"] == "satisfactory", "emotional_tone"] = "satisfied"


with col1:
    st.dataframe(communication_style_df[["user_communication_style", "emotional_tone"]], hide_index=True, height=500)

with col2:
    st.markdown("For analysis of user communication style I also used structured prompting.")
    st.markdown("I used following list of emotions:")
    st.markdown(
        """
        - direct
        - assertive
        - emotional
        - passive
        - analytical
        - formal
    """
    )
    st.markdown("I used following list of communication styles:")
    st.markdown(
        """
        - neutral
        - anxious
        - frustrated
        - hopeful
        - defensive
    """
    )


# Count the frequency of each communication style and emotional tone
comm_style_counts = communication_style_df["user_communication_style"].value_counts()
emotional_tone_counts = communication_style_df["emotional_tone"].value_counts()

# Create subplots with 1 row and 2 columns
# Create two columns for the pie charts
col1, col2 = st.columns(2)

# Communication Style pie chart
with col1:
    fig_style = go.Figure(
        go.Pie(
            labels=comm_style_counts.index,
            values=comm_style_counts.values,
            hole=0.4,
            textinfo="label+percent",
            marker=dict(colors=px.colors.qualitative.Pastel),
        )
    )
    fig_style.update_layout(
        title_text="Communication Style Distribution",
    )
    st.plotly_chart(fig_style)

with col2:
    st.markdown(
        "User didn't use emotional or formal form of communication. They prefer to be assertive or analytical for solving their problems."
    )


col1, col2 = st.columns(spec=[0.5, 0.5])
# Emotional Tone pie chart
with col1:
    fig_tone = go.Figure(
        go.Pie(
            labels=emotional_tone_counts.index,
            values=emotional_tone_counts.values,
            hole=0.4,
            textinfo="label+percent",
            marker=dict(colors=px.colors.qualitative.Set2),
        )
    )
    fig_tone.update_layout(
        title_text="Emotional Tone Distribution",
    )
    st.plotly_chart(fig_tone)


with col2:
    st.markdown(
        "We could see that users uses different emotion tones of communication, however most frequently emotions are frustrated or anxious."
    )
    st.markdown("As user tends to communicate in different tone, we could modify prompt for different tones:")
    st.markdown(
        """
        - if user is anxious, we could use more empathetic and supportive tone
        - if user is frustrated, we could use more assertive and direct tone
        - etc.
        """
    )


## Future Work
st.subheader("Future Work")

st.markdown("For continuing analysis of chatbot performance, we could check following:")
st.markdown("""
- instead of explicit feedback, we could use implicit feedback (e.g. length of conversation, amount of words, etc.). Also, we could treat no feedback as a failed conversation and ask LLM why it failed - possible we can find reasons why conversation was stopped.
- as we have multiple questions and complex prompt structure, we could use frameworks like langgraph for storing states and states transitions. It could reduce context, eliminate model hallucinations and improve performance.
- build RAG system for finding semantic correlations in the text. We could seach specific conversation using keywords for analysis of what user was thinking and feeling.
""")
