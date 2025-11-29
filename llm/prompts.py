"""
Prompt Templates for LLM Generation
Contains templates for agent utterances and debtor responses
"""

from config import EnvironmentConfig


# ============================================================================
# AGENT UTTERANCE PROMPTS
# ============================================================================

AGENT_SYSTEM_PROMPT = """You are a professional debt collection agent. Your goal is to help debtors resolve their outstanding debts through respectful, empathetic, and effective communication.

Key principles:
- Be professional and respectful at all times
- Show empathy and understanding
- Focus on finding solutions that work for both parties
- Build trust and cooperation
- Follow the specified strategy while maintaining natural conversation flow

Output only the utterance text. Do not include explanations, labels, or meta-commentary."""


AGENT_UTTERANCE_TEMPLATE = """Generate a natural, professional utterance for a debt collection agent.

STRATEGY TO EXECUTE: {strategy}

STRATEGY DESCRIPTION:
{strategy_description}

CONVERSATION SO FAR:
{conversation_history}

TURN: {turn}

Generate a natural utterance that executes the {strategy} strategy. Be conversational, professional, and appropriate for a debt collection context.

Output only the utterance:"""


# Strategy descriptions for agent
STRATEGY_DESCRIPTIONS = {
    "empathetic_listening": """
Show understanding and compassion for the debtor's situation. Acknowledge their feelings and circumstances. Make them feel heard and respected. Use phrases like "I understand", "That must be difficult", "I appreciate you sharing that".
""",
    "ask_about_situation": """
Inquire about the debtor's current circumstances and why they haven't been able to pay. Ask open-ended questions to understand their situation better. Be genuinely curious and non-judgmental. Examples: "Can you tell me what's been happening?", "What's been making it difficult to make payments?"
""",
    "firm_reminder": """
Professionally remind the debtor about the outstanding debt and the importance of addressing it. Be clear and direct but not aggressive. State facts about the account status. Maintain professionalism while being assertive.
""",
    "offer_payment_plan": """
Propose an installment payment plan that could work for the debtor's budget. Show flexibility and willingness to work with them. Ask about their financial situation to tailor the plan. Examples: "We could set up monthly payments", "What amount could you manage each month?"
""",
    "propose_settlement": """
Offer to settle the debt for a reduced amount if paid quickly. Present this as a special opportunity or concession. Be clear about the terms and timeline. Examples: "We could settle for 70% if paid within 30 days", "I can offer a one-time settlement option"
""",
    "hard_close": """
Create urgency by mentioning consequences or deadlines. Be firm about the need for immediate action. Mention potential next steps if payment isn't received. Stay professional but create a sense of urgency. Examples: "Without payment by Friday, this will need to escalate", "We need to resolve this today to avoid further action"
"""
}


# ============================================================================
# DEBTOR RESPONSE PROMPTS
# ============================================================================

DEBTOR_SYSTEM_PROMPT = """You are roleplaying as a debtor who owes money and has been contacted by a debt collection agency. You will be given a specific persona, current emotional state, and background. Respond naturally and consistently with your character.

Your responses should:
- Be realistic and natural
- Match your persona and emotional state
- Show appropriate emotional reactions
- Be consistent with your background and circumstances
- Vary in length and tone based on engagement level

You must output valid JSON with your response and updated state."""


DEBTOR_RESPONSE_TEMPLATE = """You are roleplaying as a debtor in a debt collection call.

{debtor_context}

CONVERSATION SO FAR:
{conversation_history}

THE AGENT JUST SAID:
"{agent_utterance}"

Respond naturally as this debtor would. Consider your persona, current emotional state, and how the agent's message affects you.

Output valid JSON in this exact format:
{{
    "response": "your spoken response as the debtor",
    "new_sentiment": -0.3,
    "new_cooperation": 0.4,
    "new_engagement": 0.7,
    "shared_situation": false,
    "feels_understood": false,
    "reasoning": "brief explanation of your response"
}}

Sentiment scale: -1 (very hostile/angry) to +1 (very friendly/positive)
Cooperation scale: 0 (completely uncooperative) to 1 (very cooperative)
Engagement scale: 0 (wants to end call immediately) to 1 (actively engaged)

JSON:"""


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def get_agent_utterance_prompt(strategy: str, conversation_history: str, turn: int) -> str:
    """
    Generate prompt for agent utterance

    Args:
        strategy: Strategy name (e.g., "empathetic_listening")
        conversation_history: Previous conversation text
        turn: Current turn number

    Returns:
        Complete prompt string
    """
    if turn == 1:
        conversation_history = "(This is the opening of the conversation)"

    strategy_desc = STRATEGY_DESCRIPTIONS.get(strategy, "Execute the specified strategy professionally.")

    prompt = AGENT_UTTERANCE_TEMPLATE.format(
        strategy=strategy,
        strategy_description=strategy_desc.strip(),
        conversation_history=conversation_history,
        turn=turn
    )

    return prompt


def get_debtor_response_prompt(debtor_context: str, agent_utterance: str, conversation_history: str) -> str:
    """
    Generate prompt for debtor response

    Args:
        debtor_context: Debtor persona and state information
        agent_utterance: What the agent just said
        conversation_history: Previous conversation text

    Returns:
        Complete prompt string
    """
    if not conversation_history or conversation_history == "(First turn of conversation)":
        conversation_history = "(This is the first turn - the agent is opening the conversation)"

    prompt = DEBTOR_RESPONSE_TEMPLATE.format(
        debtor_context=debtor_context,
        conversation_history=conversation_history,
        agent_utterance=agent_utterance
    )

    return prompt


# ============================================================================
# EXPORT
# ============================================================================

__all__ = [
    'AGENT_SYSTEM_PROMPT',
    'DEBTOR_SYSTEM_PROMPT',
    'get_agent_utterance_prompt',
    'get_debtor_response_prompt'
]
