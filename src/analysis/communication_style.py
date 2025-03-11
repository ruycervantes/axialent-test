## Analysis of communication styles of user and bot

from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field
import pandas as pd


# Define structured output format
class ConversationAnalysis(BaseModel):
    user_communication_style: str = Field(
        description="Communication style used by the user",
        examples=["direct", "assertive", "emotional", "passive", "analytical", "formal"],
    )
    emotional_tone: str = Field(
        description="Overall emotional tone of the user's messages",
        examples=["neutral", "anxious", "frustrated", "hopeful", "defensive"],
    )
    reasoning: str = Field(description="Brief explanation for the communication style classification")


# Initialize the LLM
llm = ChatOpenAI(model="gpt-4o-mini")
parser = PydanticOutputParser(pydantic_object=ConversationAnalysis)

# Create analysis prompt
analysis_template = """You are an expert in communication analysis with a deep understanding of linguistic patterns, emotional expression, and conversational dynamics.  

### Task:  
Analyze the USER communication style and emotional tone based on the following conversation.  

### Conversation:  
{conversation}  

### Instructions:  
- Focus **only** on the USER's messages.  
- Identify the **dominant communication style** (e.g., direct, assertive, emotional, passive, analytical, formal, casual, etc.).  
- Determine the **overall emotional tone** (e.g., neutral, anxious, frustrated, hopeful, defensive, enthusiastic, etc.).  
- Justify your classification with specific examples from the conversation.  
- If the communication style or tone shifts, note the changes and possible reasons.  

### Output Format:  
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
        conversation_text: str = extract_user_conversation_text(row)

        try:
            analysis = analyze_conversation(conversation_text)
            output_dict = {
                "conversation_id": conversation_id,
                "user_communication_style": analysis.user_communication_style,
                "emotional_tone": analysis.emotional_tone,
                "reasoning": analysis.reasoning,
            }
            analyses.append(output_dict)
        except Exception as e:
            print(f"Error analyzing conversation {conversation_id}: {e}")

    return pd.DataFrame(analyses)


# Helper function to extract conversation text from DataFrame row
def extract_user_conversation_text(row):
    """Extract and format conversation text from a DataFrame row."""
    messages = row["inputs"]["messages"]
    conversation_text = ""

    for msg in messages:
        if msg["role"] == "user":
            conversation_text += f"{msg['role'].upper()}: {msg['content']}\n"

    return conversation_text
