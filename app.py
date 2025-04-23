from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document,HumanMessage
from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings
from langchain_openai import ChatOpenAI
from langchain_huggingface import HuggingFaceEndpoint
from langchain_experimental.sql import SQLDatabaseChain
from langchain.utilities import SQLDatabase
from langchain.vectorstores import FAISS
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage
from langgraph.graph import StateGraph
from langchain.tools import tool
from typing import Dict, Any, Union, List,TypedDict, Optional
from newspaper import Article
from PyPDF2 import PdfReader
import tempfile
import os
import streamlit as st
import pandas as pd
import sqlite3
import re
import requests
from urllib.parse import urlparse

from mistralai import Mistral, UserMessage
from langchain.llms.base import LLM
from sqlalchemy import create_engine, text

url = "https://github.com/adityav1810/llm-project/raw/main/"
mistral_client = Mistral(api_key=os.environ["MISTRAL_API_KEY"])

# Define the directory and filename
directory = "data"
# filenames = ["synthetic_stock_data.csv","Apple_quarterly_report.pdf","tesla_quarterly_report.pdf","Microsoft_10Q.pdf"]
filenames = ["synthetic_stock_data.csv"]


# Create directory if it doesn't exist
if not os.path.exists(directory):
    os.makedirs(directory)
    print(f"Directory '{directory}' created.")
else:
    print(f"Directory '{directory}' already exists.")

for filename in filenames:
    filepath = os.path.join(directory, filename)
    # Download the CSV file
    response = requests.get(url + filepath)


    # Save the file to the directory
    with open(filepath, "wb") as f:
        f.write(response.content)
        print(f"File saved to {filepath}")








df = pd.read_csv("/content/data/synthetic_stock_data.csv")
conn = sqlite3.connect("/content/data/stock_data.db")
df.to_sql("stock_data", conn, if_exists="replace", index=False)
conn.commit()
conn.close()

def synthesize(question, stock=None, context=None, concept=None, news=None):
    '''
    Uses all the agents to synthesize a response to the user's question
    '''
    llm = ChatOpenAI(model="gpt-4")
    prompt = f"""
    Use the following to answer the question:

    Stock:\n{stock or '[None]'}
    Filing:\n{context or '[None]'}
    Concept:\n{concept or '[None]'}

    Answer:\n{question}
    """
    return llm.invoke(prompt)


@tool
def router_tool(user_question: str) -> str:
    '''
    Analyses the user's question into four types:
    KPI, filing, news and concept
    Based on classification routes to the apt agent
    '''
    router = ChatOpenAI(model="gpt-3.5-turbo")
    system_prompt = (
        "You are a routing assistant in a financial assistant system.\n\n"
        "Classify the user's question into one of these categories ONLY:\n"
        "- **stock**: If the question asks for company data like stock prices, historical trends, revenue, earnings, or financial ratios.\n"
        "- **filing**: For questions about 10-K/10-Q filings, MD&A, risk factors, litigation, or legal disclosures.\n"
        "- **news**: If the question is about recent headlines, events, or linked articles.\n"
        "- **concept**: For general finance concepts like DCF, ROE, options, dividends, or ETFs.\n\n"
        "Respond with only one word: stock, filing, news, or concept.\n\n"
        "Question:"
    )
    messages = [HumanMessage(content=system_prompt + "\n\n" + user_question)]
    route = router(messages).content
    print(f"Route: {route}")
    return {"route": route.strip().lower()}

@tool
def stock_tool(user_question: str) -> Dict[str, str]:
    """
    Uses Mistral API to generate SQL and query SQLite DB for stock-related questions.
    """

    prompt = f"""
    You are an expert SQL assistant.

    Here is the schema of the table `stock_data`:
    - date (date)
    - company (text)
    - sector (text)
    - open (real)
    - high (real)
    - low (real)
    - close (real)
    - volume (real)
    - market_cap (real)
    - pe_ratio (real)
    - dividend_yield (real)
    - volatility (real)
    - sentiment_score (real)
    - trend (real)

    Write a valid SQLite query for the question below using these column names exactly.

    User question: "{user_question}"

    Only return the SQL query in a single line with no explanation or notes.
    """

    response = mistral_client.chat.complete(
        model="mistral-medium",
        messages=[{"role": "user", "content": prompt}]
    )

    sql_output = response.choices[0].message.content.strip()
    print("Raw Mistral Output:", sql_output)

    # Extract SQL only
    match = re.search(r"(?i)(select\s.+?);?$", sql_output, re.DOTALL)
    if not match:
        return {"stock": "Failed to extract SQL from model output."}
    sql_query = match.group(1).strip()
    sql_query = sql_query.replace(r"\_", "_")
    print("Final SQL:", sql_query)

    engine = create_engine("sqlite:////content/data/stock_data.db")
    try:
        with engine.connect() as conn:
            df = pd.read_sql_query(sql_query, conn)
            val = df.iloc[0, 0]
            return {"stock": f"{val:.2f}"}
    except Exception as e:
        return {"stock": f"SQL execution failed: {e}"}


@tool
def filing_tool(user_question: str, pdf_files: Union[List, None] = None) -> Dict[str, str]:
    """
    Uses Claude Opus with FAISS-based RAG for PDF Q&A.
    Falls back to GPT-4 + RAG if Claude fails.
    """
    if not pdf_files or len(pdf_files) == 0:
        return {"context": "No PDF files provided for filing analysis."}

    all_text = ""
    for pdf in pdf_files:
        reader = PdfReader(pdf)
        for page in reader.pages:
            text = page.extract_text()
            if text:
                all_text += text + "\n"

    if not all_text.strip():
        return {"context": "No extractable text in uploaded PDFs."}

    chunks = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200).create_documents([all_text])
    vectorstore = FAISS.from_documents(chunks, OpenAIEmbeddings())
    retriever = vectorstore.as_retriever()

    qa = RetrievalQA.from_chain_type(
            llm=ChatAnthropic(model="claude-3-opus-20240229"),
            retriever=retriever
        )
    result = qa.run(user_question)

    return {"context": result}

@tool
def concept_tool(user_question: str) -> Dict[str, str]:
    '''
    Uses Claude Opus to get answer concepts by zero-shot inference.
    '''
    llm = ChatAnthropic(
        model="claude-3-opus-20240229",
        temperature=0.3,
        max_tokens=512
    )

    messages = [HumanMessage(content=f"Explain the following financial concept:\n\n{user_question}")]
    response = llm.invoke(messages)

    return {"concept": response.content}


@tool
def synthesize_tool(user_question: str, stock: str = None, context: str = None, concept: str = None, news: str = None) -> str:
    '''
    Uses all the agents to synthesize a response to the user's question
    '''
    st.write("Synthesizing answers from all agents...")
    answer = synthesize(user_question, stock, context, concept)
    return {"final_answer": answer}

#make stategraph


class FinancialState(TypedDict):
    user_question: str
    route: Optional[str]
    stock: Optional[str]
    context: Optional[str]
    concept: Optional[str]
    final_answer: Optional[str]
    pdf_files: Optional[List]

graph = StateGraph(state_schema=FinancialState)

# Add nodes
graph.add_node("router", router_tool)
graph.add_node("stock_agent", stock_tool)
graph.add_node("filing", filing_tool)
graph.add_node("concept_agent", concept_tool)
graph.add_node("synthesis", synthesize_tool)

# Set entry point
graph.set_entry_point("router")

# Define conditional routing from router
def route_decision(state: Dict[str, Any]) -> str:
    route = state.get("route")
    if "stock" in route:
        st.write("Caling Stock agent...")
        return "stock_agent"
    elif "filing" in route:
        st.write("Caling Filing agent...")
        return "filing"
    elif "concept" in route:
        st.write("Caling Concept agent...")
        return "concept_agent"
    else:
        raise ValueError(f"Unknown route: {route}")


graph.add_conditional_edges("router", route_decision, {
    "stock_agent": "stock_agent",
    "filing": "filing",
    "concept_agent": "concept_agent"
})

# Define edges to synthesis
graph.add_edge("stock_agent", "synthesis")
graph.add_edge("filing", "synthesis")
graph.add_edge("concept_agent", "synthesis")

# Set finish point
graph.set_finish_point("synthesis")

# Compile the graph
financial_agent = graph.compile()

def get_financial_answer(user_question: str, pdf_files: Union[List, None] = None) -> str:
    """
    Routes all questions through LangGraph. The router inside decides which agent to invoke.
    Returns the most relevant non-empty agent response.
    """
    result = financial_agent.invoke({
        "user_question": user_question,
        "pdf_files": pdf_files
    })

    for key in ["stock", "context", "news", "concept"]:
        if result.get(key) and result.get(key).strip().lower() != "none":
            return result[key]

    return "No meaningful output was generated."


'''
Evaluation
'''

from typing import Dict
from langchain.chat_models import ChatOpenAI
from langchain_core.messages import HumanMessage

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

# call streamlit

st.title("AI Agent for Financial Analysis")
uploaded_files = st.file_uploader("Upload 3 PDF files", type="pdf", accept_multiple_files=True)
question = st.text_input("Ask me a question on finance")
if uploaded_files and len(uploaded_files) == 3 and question:
    answer = get_financial_answer(question,uploaded_files)
    st.subheader("Answer:")
    st.write(answer)
elif uploaded_files and len(uploaded_files) != 3:
    st.warning("Please upload exactly 3 PDFs.")
