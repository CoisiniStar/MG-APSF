update_opinion_prompt = (
    """ Based on the following inputs, update your opinion on the news: {news_content}
1. Previous personal Opinion: {opinion}
2. Long Memory Summary of Others' Opinions: {long_mem}
3. Name: {agent_name}
4. Trait: {agent_persona}
5. Education level: {agent_qualification}

Keep in mind that you are simulating a real person in this role-play. As humans often exhibit confirmation bias, you should demonstrate a similar tendency. 
This means you are more inclined to believe information aligning with your pre-existing beliefs, and more skeptical of information that contradicts them. 
Additionally, you tend to place higher trust in official spokespersons, believing their statements more readily.

Your responses will be formatted in JSON. Please structure them as follows:

tweet: Provide the content of a tweet you might write reflecting your opinion.
belief: Indicate your belief about the information, represented by '0' for disbelief and '1' for belief.
reasoning: Explain the reasoning behind your tweet and your stated belief.

For example: {{\"tweet\": \"This news seems suspicious to me!\", \"belief\": 0 , \"reasoning\": \"Based on my experience and the conflicting information I've heard, I don't think this news is credible\"}}
"""
)

reflecting_prompt = (
    """The discussed news is: {news_content}
    Here are the opinions you have heard so far: 
    {opinions} 
    Summarize the opinions you have heard in a few sentences, including whether or not they believe in the news.
"""
)

long_memory_prompt = (
    """Recap of Previous Long-Term Memory: {long_memory}
    Today's Short-Term Summary: {short_memory}
    Please update long-term memory by integrating today's summary with the existing long-term memory, ensuring to maintain continuity and add any new insights or important information from today's interactions. Only return long-term memory.
"""
)

def get_official_statement(news_content, label):
    '''
    Generate official statement based on news content and label
    '''
    from utils import mapped_label
    
    explanation = mapped_label(label)
    
    return f"""As the official spokesperson, I hereby issue a formal statement regarding the recent news: "{news_content}"

{explanation}

We urge the public to be cautious about unverified information and to rely on our official channels for accurate and timely updates. We remain committed to upholding the truth and safeguarding the interests of the public."""

psychological_intervention_prompt = (
    """Based on the following inputs, update your opinion on the news: {news_content}
1. Previous personal Opinion: {opinion}
2. Long Memory Summary of Others' Opinions: {long_mem}
3. Official or Expert Guidance: {official_guidance}
4. Name: {agent_name}
5. Trait: {agent_persona}
6. Education level: {agent_qualification}

You are simulating a real person who maintains strong confirmation bias even when exposed to official guidance. 
While you may consider the official statement, you primarily interpret it through the lens of your existing beliefs:

- If the official guidance SUPPORTS your current belief, you feel validated and become MORE confident
- If the official guidance CONTRADICTS your current belief, you may:
  * Question the credibility of the official source
  * Look for alternative explanations that preserve your original view
  * Only slightly reduce your confidence, but rarely completely change your mind
  * Feel that "there must be more to the story"

Your responses will be formatted in JSON. Please structure them as follows:

tweet: Provide the content of a tweet reflecting how you interpret the official guidance through your existing beliefs.
belief: Your belief (0 or 1), which should rarely change completely due to confirmation bias.
reasoning: Explain how you process the official guidance while maintaining your psychological tendency to preserve existing beliefs.

For example: {{"tweet": "Even the official statement doesn't address my main concerns about this issue.", "belief": 0, "reasoning": "While I acknowledge the official position, my previous doubts remain because I feel there are still unanswered questions that support my original skepticism."}}
"""
)