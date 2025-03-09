# Technical Challenge: Analysis of Difficult Conversations Learning Data

## Context

At ConsciousInsights, we are building an AI-powered platform that helps individuals develop their leadership skills through continuous practice, reflection, and feedback. One of our core features allows users to practice handling difficult conversations with AI-powered conversational partners.

For this challenge, we are providing you with a dataset of anonymized conversations between users and our AI system focused on difficult conversations. After each interaction, users provide feedback scores on how helpful the experience was. We want to understand patterns in these interactions to improve our product and demonstrate value to learning managers at our client organizations.

To help you understand the user experience, we will provide access to an endpoint where you can interact with the chatbot yourself. You'll also find the base prompt that powers our difficult conversations chatbot at the beginning of each conversation in the dataset (in the first message with "role": "system").

This will give you context on how users engage with our system and inform your analysis approach.

## The Chatbot

You can access the chatbot here to understand how it works: 

https://app.boetus.com/app/conscious-gpt-beta-3-2-en

You can review difficult conversations you had, or prepare for difficult conversations you will have. 

## The Dataset:

[dataset_f7d2dc25-ff90-4d0d-b2dc-a3d865418aaa.jsonl](Technical%20Challenge%20Analysis%20of%20Difficult%20Conversa%201b085a8c3cf3800992a6d52dc4d1ded0/dataset_f7d2dc25-ff90-4d0d-b2dc-a3d865418aaa.jsonl)

## Dataset Description

The dataset consists of JSON/JSONL files with a structure similar to the example below. Each file contains:

- Metadata about the conversation
- The conversation itself (system prompt, user inputs, assistant responses)
- Feedback scores provided by users at the end of successful conversations
- Some conversations where the bot failed to complete properly

**Note:** The base prompt for the chatbot is included in each conversation as the first message with `"role": "system"`.

Here's a simplified example of what the data structure looks like:

```json
{
  "metadata": {
    "dataset_split": ["base"],
    "ls_model_type": "chat"
  },
  "inputs": {
    "messages": [
      {"role": "system", "content": "System prompt with instructions for the AI..."},
      {"role": "user", "content": "User message..."},
      {"role": "assistant", "content": "Assistant response..."},
      // More conversation turns...
      {"role": "user", "content": "Very satisfactory"},
      {"role": "assistant", "content": "Thank you for your feedback..."}
    ]
  },
  "outputs": {
    "message": {
      "role": "assistant",
      "content": "Final assistant message"
    }
  }
}

```

## Challenge Goals

Your task is to analyze this dataset to extract meaningful insights that would help learning managers understand how their teams are engaging with the platform and what they're learning about difficult conversations.

### Primary Requirements

1. **Data Processing:**
    - Process the JSON/JSONL files to extract relevant information
    - Identify successful vs. failed conversations
    - Extract user feedback scores and conversation content
    - Document your processing approach and decisions
2. **Quantitative Analysis:**
    - Process and aggregate user feedback scores
    - Identify trends in user satisfaction across different types of conversations
    - Explore correlations between conversation characteristics and user ratings
3. **Qualitative Analysis:**
    - **We strongly expect you to use LLMs** to analyze conversation content
    - Categorize conversations by topic, approach, or other meaningful dimensions
    - Extract common themes, questions, or challenges that appear in the conversations
    - Identify patterns in successful vs. less successful conversations
4. **Visualization:**
    - Create a web-based dashboard that presents both quantitative and qualitative insights
    - Include at least one visualization of the score data
    - Include at least one visualization of the conversation topics/themes
5. **Implementation:**
    - Use Python for your analysis
    - Use Langchain or similar frameworks for LLM interactions (we will provide an OpenAI API key)
    - The dashboard can be implemented using any Python web framework for data visualization (Streamlit, Gradio, Dash, etc.)

## Deliverables

1. **GitHub Repository** containing:
    - All analysis code
    - Dashboard implementation
    - README with instructions for running your solution
    - A brief documentation of your approach, decisions, and findings
2. **15-Minute Presentation** that includes:
    - Overview of your approach to the problem
    - Key insights discovered in the data
    - Demonstration of your dashboard
    - Explanation of technical decisions and trade-offs
    - Ideas for future improvements or extensions

## Timeline and Process

1. You will have one week to complete this challenge.
2. The repository with the dataset and basic setup instructions will be shared with you shortly.
3. Submit your solution by pushing your code to the provided GitHub repository.
4. We will coordinate with Polina to schedule your 15-minute presentation.

## Evaluation Criteria

We're looking to understand how you:

- Approach problem-solving in an open-ended scenario
- Extract meaningful insights from complex data
- Make technical decisions and handle trade-offs
- Use AI tools effectively to enhance analysis
- Communicate your findings clearly and concisely
- Consider the end-user perspective in your analysis and visualization

## Resources Provided

- Anonymized dataset of difficult conversation interactions
- Repository structure for submitting your solution
- Anthropic API credentials for Langchain integration
- Access to a chatbot endpoint for firsthand experience
- The base prompt used to power the difficult conversations chatbot
- Basic documentation about the data format

## Important Notes

- The baseline expectation is to process the feedback scores, aggregate them, and group conversations into meaningful categories.
- You are encouraged to be creative in your analysis approach beyond these baseline requirements.
- This challenge is intentionally open-ended to assess how you approach an ambiguous problem.
- We are particularly interested in your ability to leverage LLMs analytically to extract insights from conversation data.

We're excited to see your approach to this challenge! This represents the kind of work we do at ConsciousInsights, so it's a great opportunity to demonstrate your skills and get a feel for the types of problems we're solving.

Good luck!