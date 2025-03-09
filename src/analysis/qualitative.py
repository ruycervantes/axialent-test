from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field
from typing import List
import pandas as pd


# Define structured output format
class ConversationAnalysis(BaseModel):
    main_topic: str = Field(description="The primary topic of the conversation")
    subtopics: List[str] = Field(description="List of subtopics discussed")
    communication_style: str = Field(description="Style of communication used")
    key_challenges: List[str] = Field(description="Main challenges or pain points identified")
    success_factors: List[str] = Field(description="Elements that contributed to conversation success")
    sentiment_score: int = Field(
        description="Overall sentiment score (1 to 5, where 1 is very negative and 5 is very positive)"
    )
    user_engagement: int = Field(description="Level of user engagement (1 to 5, where 1 is low and 5 is high)")
    dialog_success: int = Field(
        description="Success of the dialog in addressing the issue (0 to 1, where 0 is unsuccessful and 1 is very successful)"
    )
    user_feedback_score: int = Field(
        description="General feedback from the user (1 to 5, where 1 is very negative or no feedback and 5 is very positive)"
    )


# Initialize the LLM
llm = ChatOpenAI(model="gpt-4o-mini")
parser = PydanticOutputParser(pydantic_object=ConversationAnalysis)

# Create analysis prompt
analysis_template = """
You are an expert in analyzing difficult conversations and communication patterns.
Analyze the following conversation between a user and an AI assistant practicing difficult conversations.

Conversation:
{conversation}

Please analyze this conversation and provide a structured analysis.
Focus on identifying:
1. The main topic and any subtopics
2. The communication style used
3. Key challenges or pain points
4. Elements that made the conversation successful or unsuccessful
5. Overall sentiment (1 to 5 scale, where 1 is very negative and 5 is very positive)
6. User engagement level (1 to 5 scale, where 1 is low and 5 is high)
7. Dialog success in addressing the issue (0 to 1 scale, where 0 is unsuccessful and 1 is very successful)
8. User feedback score (1 to 5 scale, where 1 is very negative or no feedback and 5 is very positive)

{format_instructions}
"""

prompt = ChatPromptTemplate.from_template(
    template=analysis_template, partial_variables={"format_instructions": parser.get_format_instructions()}
)


def analyze_conversation(conversation_text: str) -> ConversationAnalysis:
    """Analyze a single conversation using the LLM."""
    messages = prompt.format_messages(conversation=conversation_text)
    response = llm.invoke(messages)
    return parser.parse(response.content)


# Function to analyze conversation patterns across multiple conversations
def analyze_conversation_patterns(conversations_df: pd.DataFrame):
    """Analyze patterns across multiple conversations."""
    analyses = []

    for conversation_id, row in conversations_df.iterrows():
        # Extract conversation text from the DataFrame
        conversation_text: str = extract_conversation_text(row)

        try:
            analysis = analyze_conversation(conversation_text)
            output_dict = {
                "conversation_id": conversation_id,
                "analysis": analysis,
            }
            output_dict.update(analysis.model_dump())
            analyses.append(output_dict)
        except Exception as e:
            print(f"Error analyzing conversation {conversation_id}: {e}")

    return pd.DataFrame(analyses)


# Function to identify common themes across conversations
def identify_themes(analyses_df: pd.DataFrame) -> dict:
    """Identify common themes and patterns from analyzed conversations."""

    theme_prompt = """
    Review these conversation analyses and identify:
    1. Most common topics and subtopics
    2. Patterns in successful conversations
    3. Common challenges
    4. Effective communication strategies
    5. Patterns in user engagement and dialog success

    Analyses:
    {analyses}
    """

    # Prepare analyses summary
    analyses_text = analyses_df["analysis"].to_string()

    # Get theme analysis from LLM
    response = llm.invoke(theme_prompt.format(analyses=analyses_text))

    return response.content


# Helper function to extract conversation text from DataFrame row
def extract_conversation_text(row):
    """Extract and format conversation text from a DataFrame row."""
    messages = row["inputs"]["messages"]
    conversation_text = ""

    for msg in messages:
        if msg["role"] != "system":  # Skip system prompts
            conversation_text += f"{msg['role'].upper()}: {msg['content']}\n\n"

    return conversation_text
