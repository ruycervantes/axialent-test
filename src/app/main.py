import streamlit as st
import pandas as pd

import plotly.express as px

from collections import Counter

st.title("Axialent Test Challeng task")

df_path = "data/output_df.csv"
df = pd.read_csv(df_path).drop(columns=["analysis"])

quantative_df_path = "data/quantative_df.csv"
quantative_df = pd.read_csv(quantative_df_path)


df = pd.concat([df, quantative_df], axis=1)

df.challenge_clusters = df.challenge_clusters.apply(pd.eval)
df.success_factor_clusters = df.success_factor_clusters.apply(pd.eval)

st.dataframe(df)


def show_conversation_statistics(df: pd.DataFrame):
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Conversation Length")
        fig = px.histogram(df, x="conversation_length", title="Conversation Length Distribution", nbins=10)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("Number of User Words")
        fig = px.histogram(df, x="number_of_user_words", title="Number of User Words Distribution", nbins=10)
        st.plotly_chart(fig, use_container_width=True)


def create_score_visualization(df: pd.DataFrame):
    # Create two columns for the visualizations
    col1, col2 = st.columns(2)

    # First column for sentiment score
    with col1:
        st.subheader("Sentiment Score Distribution")
        fig_dist = px.histogram(
            df, x="sentiment_score", title="Distribution of Sentiment Scores", nbins=5, labels=[1, 2, 3, 4, 5]
        )
        st.plotly_chart(fig_dist, use_container_width=True)

    # Second column for user engagement
    with col2:
        st.subheader("User engagement scores")
        fig_engagement = px.histogram(
            df, x="user_engagement", title="Distribution of Engagement Scores", nbins=5, labels=[1, 2, 3, 4, 5]
        )
        st.plotly_chart(fig_engagement, use_container_width=True)


def show_most_common_main_topic_clusters(df: pd.DataFrame):
    # Count occurrences of each main topic
    topic_counts = Counter(df["topic_cluster"])

    # Get the top 10 most common topics
    # most_common_topics = topic_counts.most_common(10)
    most_common_topics = topic_counts.most_common()

    # Create a dataframe for plotting
    topics_df = pd.DataFrame(most_common_topics, columns=["Topic Cluster", "Count"])

    # Create the bar chart
    st.subheader("Topic Clusters")
    fig = px.bar(
        topics_df,
        x="Topic Cluster",
        y="Count",
        title="Most Common Topic Clusters",
        color="Count",
        color_continuous_scale=px.colors.sequential.Viridis,
    )

    fig.update_layout(xaxis_title="Topic Cluster", yaxis_title="Frequency")
    st.plotly_chart(fig, use_container_width=True)


def show_most_common_challenges(df: pd.DataFrame):
    # Count occurrences of each challenge
    challenge_counts = Counter(df["challenge_clusters"].explode())

    # Get the top 10 most common challenges
    most_common_challenges = challenge_counts.most_common(10)

    # Create a dataframe for plotting
    challenges_df = pd.DataFrame(most_common_challenges, columns=["Challenge Cluster", "Count"])

    # Create the bar chart
    st.subheader("Most Frequent Challenges")
    fig = px.bar(
        challenges_df,
        x="Challenge Cluster",
        y="Count",
        title="Top 10 Most Common Challenges",
        color="Count",
        color_continuous_scale=px.colors.sequential.Viridis,
    )

    fig.update_layout(xaxis_title="Challenge Cluster", yaxis_title="Frequency")
    st.plotly_chart(fig, use_container_width=True)


def show_most_common_success_factors(df: pd.DataFrame):
    # Count occurrences of each challenge
    challenge_counts = Counter(df["success_factor_clusters"].explode())

    # Get the top 10 most common challenges
    most_common_challenges = challenge_counts.most_common(10)

    # Create a dataframe for plotting
    challenges_df = pd.DataFrame(most_common_challenges, columns=["Success Factor Cluster", "Count"])

    # Create the bar chart
    st.subheader("Most Frequent Success Factors")
    fig = px.bar(
        challenges_df,
        x="Success Factor Cluster",
        y="Count",
        title="Success Factors",
        color="Count",
        color_continuous_scale=px.colors.sequential.Viridis,
    )

    fig.update_layout(xaxis_title="Success Factor Cluster", yaxis_title="Frequency")
    st.plotly_chart(fig, use_container_width=True)


def show_feedback_statistics(quantative_df: pd.DataFrame):
    st.subheader("Feedback Statistics")

    # Calculate feedback response percentages
    feedback_percentages = [
        len(quantative_df["feedback_Q1"].dropna()) / len(quantative_df) * 100,
        len(quantative_df["feedback_Q2"].dropna()) / len(quantative_df) * 100,
        len(quantative_df["feedback_Q3"].dropna()) / len(quantative_df) * 100,
    ]

    # Create dataframe for plotting
    feedback_df = pd.DataFrame({"Question": ["Q1", "Q2", "Q3"], "Percentage": feedback_percentages})

    # Create the bar chart with Plotly
    fig = px.bar(
        feedback_df,
        x="Question",
        y="Percentage",
        title="Percentage of Feedback Responses by Question",
        color="Percentage",
        color_continuous_scale=px.colors.sequential.Viridis,
        text=[f"{val:.1f}%" for val in feedback_percentages],
    )

    fig.update_layout(xaxis_title="Question", yaxis_title="Percentage of Responses", yaxis=dict(range=[0, 100]))

    fig.update_traces(textposition="outside")

    # Display the plot
    st.plotly_chart(fig, use_container_width=True)


def show_themes_where_feedback_was_given_and_not_given(df: pd.DataFrame, quantative_df: pd.DataFrame):
    st.subheader("Themes where feedback was given and not given")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Themes where feedback was given")

        df_with_feedback = df[quantative_df["feedback_Q1"].notna()]

        most_common_topics = Counter(df_with_feedback["topic_cluster"]).most_common()

        # Create a dataframe for plotting
        topics_df = pd.DataFrame(most_common_topics, columns=["Topic Cluster", "Count"])

        # Create the bar chart
        fig = px.bar(
            topics_df,
            x="Topic Cluster",
            y="Count",
            title="Most Common Topic Clusters",
            color="Count",
            color_continuous_scale=px.colors.sequential.Viridis,
        )

        fig.update_layout(xaxis_title="Topic Cluster", yaxis_title="Frequency")
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("Themes where feedback was not given")
        df_without_feedback = df[quantative_df["feedback_Q1"].isna()]

        most_common_topics = Counter(df_without_feedback["topic_cluster"]).most_common()

        # Create a dataframe for plotting
        topics_df = pd.DataFrame(most_common_topics, columns=["Topic Cluster", "Count"])

        # Create the bar chart
        fig = px.bar(
            topics_df,
            x="Topic Cluster",
            y="Count",
            title="Most Common Topic Clusters",
            color="Count",
            color_continuous_scale=px.colors.sequential.Viridis,
        )

        fig.update_layout(xaxis_title="Topics Cluster", yaxis_title="Frequency")
        st.plotly_chart(fig, use_container_width=True)


show_conversation_statistics(df)
create_score_visualization(df)
show_most_common_main_topic_clusters(df)
show_most_common_challenges(df)
show_most_common_success_factors(df)
show_feedback_statistics(quantative_df)
show_themes_where_feedback_was_given_and_not_given(df, quantative_df)
