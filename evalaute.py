'''
Evaluation
'''

from typing import Dict
from langchain.chat_models import ChatOpenAI
from langchain_core.messages import HumanMessage
from app import *
evaluator_llm = ChatOpenAI(model="gpt-4")

def evaluate_agent_output(user_question: str, agent_response: str, evidence: str = None) -> Dict[str, str]:
    if evidence and evidence.strip().lower() != "none":
        prompt = f"""
        You are an expert evaluator of AI-generated responses.
        Evaluate the following AI response to a user question using the provided supporting evidence.

        ### User Question:
        {user_question}

        ### AI Response:
        {agent_response}

        ### Supporting Evidence:
        {evidence}

        Evaluate on the following criteria:
        1. **Relevance** (0–10): Does the AI response directly and completely answer the user's question?
        2. **Hallucination** (0–10): Does the AI add information that is not clearly supported by the evidence?
        3. **Short Explanation**: Describe any mismatches, missing details, or unsupported claims.

        Respond in the following format:
        Relevance Score: X
        Hallucination Score: Y
        Explanation: ...
        """
    else:
        prompt = f"""
        You are an expert evaluator of AI-generated responses.
        Evaluate the following AI response to a user question.

        ### User Question:
        {user_question}

        ### AI Response:
        {agent_response}

        Evaluate on the following criteria:
        1. **Relevance** (0–10): Does the AI response directly and completely answer the user's question?
        2. **Hallucination** (0–10): Does the AI include fabricated or irrelevant information?
        3. **Short Explanation**: Mention if the answer is vague, off-topic, or unsupported by the question.

        Respond in the following format:
        Relevance Score: X
        Hallucination Score: Y
        Explanation: ...
        """

    result = evaluator_llm.invoke([HumanMessage(content=prompt)])
    return result.content

evaluator_llm = ChatOpenAI(model="gpt-4")

def evaluate_sql_query(user_question: str, sql_query: str, schema_description: str) -> dict:
    """
    Uses GPT-4 to evaluate the SQL query generated for a user's question.

    Returns:
        dict with relevance_score, hallucination_score, and explanation.
    """
    prompt = f"""
    You are an expert SQL evaluator.

    Evaluate the following SQL query based on how well it answers the user's question,
    and whether it introduces any hallucinated columns, logic, or structure.

    ### Table Schema:
    {schema_description}

    ### User Question:
    {user_question}

    ### SQL Query:
    {sql_query}

    Rate the query on:
    - Relevance (0–10): Does the SQL reflect the question's intent accurately?
    - Hallucination (0–10): Does the SQL include any invented fields, tables, or unsupported logic?

    Respond in this format:
    Relevance Score: X
    Hallucination Score: Y
    Explanation: ...
    """

    result = evaluator_llm.invoke([HumanMessage(content=prompt)])
    return result.content
'''
Example code to call evaluator for filing agent.
answer = get_financial_answer("Extract key highlights from Microsoft’s MD&A section.", pdf_files=pdf_files)
evaluate_agent_output("Extract key highlights from Microsoft’s MD&A section.", answer['context'], answer['evidence'])
'''
questions = [
    "Summarize Tesla’s key highlights from the latest MD&A section.",
    "What are Microsoft’s core business segments and how did each perform this quarter?",
    "What guidance did Apple provide for future revenue growth?",
    "How did Tesla’s operating income change compared to the previous quarter?",
    "What major trends are discussed in Microsoft’s industry overview?",
    "What does Apple cite as the primary drivers of growth this quarter?",
    "How did Microsoft’s cost of revenue evolve?",
    "What are Tesla’s R&D priorities this quarter?",
    "Summarize the fiscal outlook Apple provided in its latest report.",
    "What updates did Microsoft give on Azure’s performance?",
    "What risks did Tesla highlight related to raw material sourcing?",
    "What litigation risks does Apple mention?",
    "How does Microsoft describe its competitive threats in cloud?",
    "Did Apple disclose any cybersecurity incidents?",
    "What financial risks are associated with Tesla’s supply chain?",
    "What legal proceedings involve Microsoft this quarter?",
    "Did Tesla update its risk disclosure around vehicle recalls?",
    "How does Apple assess currency exchange rate risk?",
    "What geopolitical risks are flagged in Microsoft’s 10-Q?",
    "Has Tesla added any new risk factors this quarter?",
    "What was the revenue from Apple’s iPhone segment this quarter?",
    "How did Tesla Energy perform relative to prior quarters?",
    "How did Microsoft's Productivity and Business Processes segment perform?",
    "What is the breakdown of Apple’s Services revenue?",
    "What trends does Tesla report for its automotive regulatory credits?",
    "What was the growth rate for Microsoft’s Intelligent Cloud segment?",
    "How does Apple categorize its Wearables and Home Accessories revenue?",
    "Has Tesla provided updated unit delivery numbers?",
    "How much revenue did Microsoft’s Surface line generate?",
    "How did Apple’s Mac revenue compare to the previous quarter?",
    "What was Apple’s gross margin this quarter?",
    "How did Tesla’s cost of goods sold change?",
    "Did Microsoft disclose any major cost optimizations?",
    "What were Apple’s R&D and SG&A expenses?",
    "What does Tesla attribute its margin compression to?",
    "What is Microsoft’s reported operating margin?",
    "How much cash on hand does Apple report?",
    "Did Tesla make any notable capital expenditures?",
    "What were Microsoft’s investing activities this quarter?",
    "How much did Apple return to shareholders via dividends or buybacks?",
    "What future plans did Tesla outline in its outlook?",
    "What new product directions does Apple hint at?",
    "Does Microsoft indicate further expansion into AI or cloud services?",
    "What are Tesla’s global expansion plans?",
    "How does Apple describe its long-term sustainability strategy?",
    "Did Tesla update its depreciation schedule or accounting methods?",
    "What are Apple’s off-balance-sheet arrangements?",
    "What employee-related disclosures does Microsoft provide?",
    "How does Tesla report on regulatory or ESG compliance?",
    "Does Apple disclose any material changes in leadership or governance?"
]



batch_results = []
for question in questions:
    result = get_financial_answer(question)
    if "context" not in result or result["context"].strip().lower() == "none":
      continue
    evaluation = evaluate_agent_output(question, result["context"], result["evidence"])
    batch_results.append({
        "question": question,
        "answer": result["context"],
        "evidence": result["evidence"],
        "evaluation": evaluation
    })

df_eval = pd.DataFrame(batch_results)
def parse_scores(evaluation_text):
    relevance = hallucination = None
    for line in evaluation_text.split("\n"):
        if "Relevance Score" in line:
            relevance = int(re.search(r'\d+', line).group())
        elif "Hallucination Score" in line:
            hallucination = int(re.search(r'\d+', line).group())
    return relevance, hallucination

df_eval[["relevance_score", "hallucination_score"]] = df_eval["evaluation"].apply(
    lambda x: pd.Series(parse_scores(x))
)

# Compute averages
average_relevance = df_eval["relevance_score"].mean()
average_hallucination = df_eval["hallucination_score"].mean()
def parse_scores(evaluation_text):
    relevance = hallucination = None
    for line in evaluation_text.split("\n"):
        if "Relevance Score" in line:
            relevance = int(re.search(r'\d+', line).group())
        elif "Hallucination Score" in line:
            hallucination = int(re.search(r'\d+', line).group())
    return relevance, hallucination

df_eval[["relevance_score", "hallucination_score"]] = df_eval["evaluation"].apply(
    lambda x: pd.Series(parse_scores(x))
)

# Compute averages
average_relevance = df_eval["relevance_score"].mean()
average_hallucination = df_eval["hallucination_score"].mean()

results = {
    "Average Relevance Score": round(average_relevance, 2),
    "Average Hallucination Score": round(average_hallucination, 2)
}
with open('results.json', 'w') as f:
    json.dump(results, f)
