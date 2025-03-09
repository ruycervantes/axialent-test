from langchain.prompts import ChatPromptTemplate
from typing import List, Dict
from langchain_openai import ChatOpenAI
import pandas as pd


class LLMTopicClusterer:
    def __init__(self, llm):
        self.llm = llm

        self.clustering_prompt = ChatPromptTemplate.from_template("""
        You are an expert at analyzing and categorizing conversation topics.
        
        Task: Group the following items into 5-7 meaningful clusters. Each item should be assigned to exactly one cluster.
        
        Items to cluster:
        {items}
        
        Provide your response in the following format:
        Cluster 1: [Brief cluster description]
        - item1
        - item2
        
        Cluster 2: [Brief cluster description]
        - item3
        - item4
        
        And so on.
        
        Make sure every item from the input list is included exactly once.
        """)

    def cluster_items(self, items: List[str]) -> Dict[str, List[str]]:
        """Cluster a list of items using LLM"""
        # Format items for the prompt
        items_text = "\n".join([f"- {item}" for item in items])

        # Get clustering from LLM
        messages = self.clustering_prompt.format_messages(items=items_text)
        response = self.llm.invoke(messages)

        # Parse the response into a dictionary
        clusters = {}
        current_cluster = None

        for line in response.content.split("\n"):
            line = line.strip()
            if not line:
                continue

            if line.startswith("Cluster"):
                current_cluster = line.split(":")[1].strip()
                clusters[current_cluster] = []
            elif line.startswith("- "):
                if current_cluster is not None:
                    clusters[current_cluster].append(line[2:])

        return clusters


def analyze_conversations_with_llm_clustering(analyses_df: pd.DataFrame, llm: ChatOpenAI | None = None):
    """Analyze conversations using LLM clustering"""

    # Initialize clusterer
    if llm is None:
        llm = ChatOpenAI(model="gpt-4o-mini")

    clusterer = LLMTopicClusterer(llm)

    # 1. Cluster main topics
    unique_topics = analyses_df["main_topic"].unique().tolist()
    topic_clusters = clusterer.cluster_items(unique_topics)

    # 2. Cluster challenges
    unique_challenges = list(
        set([challenge for challenges in analyses_df["key_challenges"] for challenge in challenges])
    )
    challenge_clusters = clusterer.cluster_items(unique_challenges)

    unique_success_factors = list(
        set(
            [success_factor for success_factors in analyses_df["success_factors"] for success_factor in success_factors]
        )
    )
    success_factor_clusters = clusterer.cluster_items(unique_success_factors)

    # Create mapping dictionaries
    topic_to_cluster = {topic: cluster_name for cluster_name, topics in topic_clusters.items() for topic in topics}

    challenge_to_cluster = {
        challenge: cluster_name for cluster_name, challenges in challenge_clusters.items() for challenge in challenges
    }

    success_factor_to_cluster = {
        success_factor: cluster_name
        for cluster_name, success_factors in success_factor_clusters.items()
        for success_factor in success_factors
    }

    # Add clustered columns to DataFrame
    analyses_df["topic_cluster"] = analyses_df["main_topic"].map(topic_to_cluster)
    analyses_df["topic_cluster"] = analyses_df["topic_cluster"].apply(lambda x: x[0] if isinstance(x, list) else x)

    # Handle challenges (which are in lists)
    analyses_df["challenge_clusters"] = analyses_df["key_challenges"].apply(
        lambda challenges: [challenge_to_cluster.get(c) or c for c in challenges]
    )

    analyses_df["success_factor_clusters"] = analyses_df["success_factors"].apply(
        lambda success_factors: [success_factor_to_cluster.get(s) or s for s in success_factors]
    )

    return analyses_df, topic_clusters, challenge_clusters, success_factor_clusters


def analyze_clustered_data(analyses_df: pd.DataFrame):
    """Analyze the clustered conversation data"""

    # Analyze by topic clusters
    topic_cluster_analysis = (
        analyses_df.groupby("topic_cluster")
        .agg({"dialog_success": ["count", "mean"], "user_feedback_score": "mean", "sentiment_score": "mean"})
        .round(2)
    )

    # Analyze by challenge clusters
    challenge_df = analyses_df.explode("challenge_clusters")
    challenge_cluster_analysis = (
        challenge_df.groupby("challenge_clusters")
        .agg({"dialog_success": ["count", "mean"], "sentiment_score": "mean"})
        .round(2)
    )

    return {"topic_cluster_analysis": topic_cluster_analysis, "challenge_cluster_analysis": challenge_cluster_analysis}


def visualize_clustered_data(analyses_df: pd.DataFrame, topic_clusters: Dict, challenge_clusters: Dict):
    """Create visualizations for clustered data"""
    import plotly.express as px

    # 1. Topic Cluster Performance
    topic_performance = (
        analyses_df.groupby("topic_cluster")
        .agg({"dialog_success": "mean", "user_feedback_score": "mean", "sentiment_score": "mean"})
        .reset_index()
    )

    fig_topics = px.bar(
        topic_performance.melt(id_vars=["topic_cluster"]),
        x="topic_cluster",
        y="value",
        color="variable",
        title="Performance by Topic Cluster",
        barmode="group",
    )

    # 2. Challenge Cluster Impact
    challenge_df = analyses_df.explode("challenge_clusters")
    challenge_impact = challenge_df.groupby("challenge_clusters")["dialog_success"].agg(["count", "mean"]).reset_index()

    fig_challenges = px.scatter(
        challenge_impact,
        x="mean",
        y="count",
        text="challenge_clusters",
        title="Challenge Cluster Impact",
        labels={"mean": "Average Dialog Success", "count": "Frequency"},
    )

    return {"topic_performance": fig_topics, "challenge_impact": fig_challenges}


#
