# AI-Powered Financial Analyst Assistant

This project is a modular AI system that uses multiple LLM-powered agents to automate key tasks in financial analysis. It leverages Prompt Engineering, Retrieval Augmented Generation (RAG), and Agent-based orchestration to generate insightful, accurate, and structured answers from stock data, company filings, financial news, and finance concepts.

---

## Project Goals

Traditional financial analysis can be time-consuming and technically challenging. Analysts spend hours digging through earnings reports, filings, and news — often without strong programming experience. This project mitigates that challenge by:

- Automating financial data extraction
- Providing natural language access to financial databases and documents
- Summarizing and synthesizing complex information across multiple sources
- Empowering analysts using LLMs with memory and reasoning abilities

---

## System Architecture

The system uses the LangGraph framework to orchestrate AI agents:

- **Router Agent (GPT-3.5)**: Classifies the user query into `stock`, `filing`, `news`, or `concept`
- **Stock Agent (Mistral)**: Generates SQL to query financial stock data
- **Filing Agent (Claude Opus)**: Performs RAG over 10-K/10-Q PDFs using FAISS
- **Concept Agent (Claude Opus)**: Explains financial concepts like DCF, ROE, etc.
- **Synthesis Agent (GPT-4)**: Synthesizes insights from the above agents into one final response

---

## 📂 Repository Structure

```bash
├── app.py             # Defines agent tools, logic, and LangGraph graph
├── run.ipynb          # Jupyter Notebook interface to run and test the application
├── evaluate.py        # Evaluation script for system performance and agent outputs
├── README.md          # Project documentation

## Instructions to run the application :

1. Run run.ipynb
2. add OpenAI keys
3. open ssh tunnel to open the website.

