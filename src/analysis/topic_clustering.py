from langchain.prompts import PromptTemplate
from langchain.output_parsers import PydanticOutputParser
from langchain_openai import OpenAI
from pydantic import BaseModel, Field
from typing import List


# First define our output structure using Pydantic
class ConversationCategory(BaseModel):
    main_topic: str = Field(description="The main topic category of the conversation")
    situation: str = Field(description="The specific situation within that topic")
    confidence: float = Field(description="Confidence score between 0 and 1")
    reasoning: str = Field(description="Brief explanation for the categorization")


# Create the parser
parser = PydanticOutputParser(pydantic_object=ConversationCategory)


CATEGORIES = {
    "Performance Related": [
        "Negative performance reviews",
        "Missed deadlines",
        "Poor quality work",
        "Productivity concerns",
        "Attendance/punctuality issues",
    ],
    "Interpersonal Conflicts": [
        "Team member conflicts",
        "Inappropriate behavior/comments",
        "Personality clashes",
        "Project approach disagreements",
        "Communication style differences",
    ],
    "Career Development": [
        "Declining promotions",
        "Explaining candidate rejection",
        "Unmet salary expectations",
        "Skill gaps discussion",
        "Career transition talks",
    ],
    "Organizational Changes": [
        "Layoff announcements",
        "Role changes",
        "Policy changes",
        "Budget cuts",
        "Merger/acquisition impacts",
    ],
    "Professional Conduct": [
        "Dress code violations",
        "Hygiene issues",
        "Company resource misuse",
        "Social media behavior",
        "Confidentiality breaches",
    ],
    "Work Life Balance": [
        "Overtime expectations",
        "Personal issues affecting work",
        "Leave requests",
        "Flexible work arrangements",
        "Return-to-office transitions",
    ],
    "Ethics Compliance": [
        "Ethical violations",
        "Compliance issues",
        "Harassment complaints",
        "Discrimination concerns",
        "Policy violations",
    ],
    "Project Management": [
        "Project failures",
        "Missed milestones",
        "Budget overruns",
        "Scope creep",
        "Resource allocation conflicts",
    ],
    "Client Relations": [
        "Client complaints",
        "Service failures",
        "Expectation management",
        "Contract disputes",
        "Price increase communications",
    ],
    "Team Dynamics": [
        "Underperforming team members",
        "Toxic behavior",
        "Leadership challenges",
        "Team restructuring",
        "Collaboration issues",
    ],
}

# Create the prompt template
template = """You are an expert in workplace communication analysis.  
Your task is to categorize the given conversation based on predefined workplace topics and situations.  


### **Categories & Situations:**  
{categories}

### **Conversation to Analyze:**  
{conversation}

### **Instructions:**  
- Identify the most relevant **main topic** and **specific situation** from the provided categories.  
- If multiple topics are discussed, select the **most dominant one**.  
- Provide a **confidence score** (0 to 1) reflecting how well the conversation matches the chosen category.  
- Offer a **concise reasoning** for the categorization.  

### **Output Format:**  
Respond in the following JSON format:  

{format_instructions}
"""

prompt = PromptTemplate(
    template=template,
    input_variables=["conversation", "categories"],
    partial_variables={"format_instructions": parser.get_format_instructions()},
)


def analyze_conversation(conversation_text: str, model: OpenAI) -> ConversationCategory:
    """
    Analyzes a conversation and returns its category, situation, and confidence score
    """
    # Format categories as a readable string
    categories_str = "\n".join(
        f"{topic}:\n" + "\n".join(f"- {situation}" for situation in situations)
        for topic, situations in CATEGORIES.items()
    )

    # Generate the prompt
    formatted_prompt = prompt.format(conversation=conversation_text, categories=categories_str)

    # Get response from the model
    response = model.invoke(formatted_prompt)

    # Parse the response into our structured format
    try:
        result = parser.parse(response.content)
        return result
    except Exception as e:
        print(f"Error parsing response: {e}")
        return None


# Example usage:
def process_conversations(conversations: List[dict], model: OpenAI):
    """
    Process a list of conversations and categorize each one
    """
    results = []

    for conv in conversations:
        # Extract the conversation text from your JSON structure
        conversation_text = extract_conversation_text(conv)

        # Analyze the conversation
        category = analyze_conversation(conversation_text, model)

        if category:
            results.append(
                {"conversation_id": conv.get("conversation_id", "unknown"), "category": category.model_dump()}
            )

    return results


def extract_conversation_text(conv: dict) -> str:
    """
    Extract conversation text from the JSON structure
    Adjust this based on your actual data structure
    """
    messages = []
    for msg in conv["inputs"]["messages"]:
        if msg["role"] != "system":  # Skip the system prompt
            messages.append(f"{msg['role'].upper()}: {msg['content']}")

    return "\n".join(messages)
