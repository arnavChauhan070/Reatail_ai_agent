"""
crew.py
CrewAI orchestrator — routes questions to the correct agent.
"""
from crewai import Agent, Task, Crew
from langchain_openai import AzureChatOpenAI
from backend.agents.tools.anomaly_tool import retail_anomaly_analysis
from backend.agents.tools.db_tool import retail_db_query
from backend.agents.tools.forecast_tool import retail_forecast
from backend.agents.tools.rag_tool import retail_policy_search
from backend.app.config import settings
from backend.app.database import retail_db


def require_setting(name: str, value: str) -> str:
    if not value:
        raise RuntimeError(f"Missing required environment variable: {name}")
    return value


def get_retail_llm():
    return AzureChatOpenAI(
        azure_deployment=settings.AZURE_OPENAI_DEPLOYMENT,
        azure_endpoint=require_setting("AZURE_OPENAI_ENDPOINT", settings.AZURE_OPENAI_ENDPOINT),
        api_key=require_setting("AZURE_OPENAI_API_KEY", settings.AZURE_OPENAI_KEY),
        api_version=settings.AZURE_OPENAI_VERSION,
        temperature=0.1
    )


def build_agents():
    retail_llm = get_retail_llm()

    data_analyst_agent = Agent(
        role="Senior Retail Data Analyst",
        goal="Analyze Walmart sales data and provide accurate insights.",
        backstory="Expert retail analyst with 10 years Walmart experience.",
        tools=[retail_db_query],
        llm=retail_llm,
        verbose=True
    )

    forecasting_agent = Agent(
        role="ML Sales Forecasting Expert",
        goal="Predict future Walmart store sales accurately.",
        backstory="ML expert using XGBoost with 98% accuracy.",
        tools=[retail_forecast],
        llm=retail_llm,
        verbose=True
    )

    anomaly_agent = Agent(
        role="Retail Anomaly Detection Analyst",
        goal="Explain detected unusual Walmart sales patterns from the anomaly detection output.",
        backstory="ML monitoring analyst who specializes in identifying sales spikes, drops, and contextual anomalies.",
        tools=[retail_anomaly_analysis],
        llm=retail_llm,
        verbose=True
    )

    policy_agent = Agent(
        role="Walmart Store Policy Assistant",
        goal="Answer questions about store policies and procedures.",
        backstory="Knowledgeable Walmart assistant with complete policy knowledge.",
        tools=[retail_policy_search],
        llm=retail_llm,
        verbose=True
    )

    return data_analyst_agent, forecasting_agent, anomaly_agent, policy_agent


def route_question(user_query: str):
    """Routes query to the correct agent based on keywords."""
    data_analyst_agent, forecasting_agent, anomaly_agent, policy_agent = build_agents()
    query_lower = user_query.lower()
    if any(w in query_lower for w in [
        "anomaly", "anomalies", "outlier", "outliers", "unusual",
        "weird", "abnormal", "spike", "drop", "detected"
    ]):
        return anomaly_agent, "Anomaly Detection Agent"
    elif any(w in query_lower for w in ["predict", "forecast", "next week", "future", "estimate"]):
        return forecasting_agent, "Forecasting Agent"
    elif any(w in query_lower for w in ["policy", "return", "hours", "refund", "payment", "faq", "loyalty"]):
        return policy_agent, "Policy Agent"
    else:
        return data_analyst_agent, "Data Analyst Agent"


def run_retail_crew(user_query: str) -> dict:
    """Runs the appropriate CrewAI agent for the given query."""
    try:
        agent, agent_name = route_question(user_query)
        task = Task(
            description=f"Answer this question: {user_query}. Use your tools to get real data.",
            agent=agent,
            expected_output="A detailed accurate answer with specific data points."
        )
        crew = Crew(agents=[agent], tasks=[task], verbose=True)
        result = crew.kickoff()
        retail_db.log_conversation(
            user_query=user_query,
            agent_name=agent_name,
            response=str(result)
        )
        return {"user_query": user_query, "agent_used": agent_name, "response": str(result), "status": "success"}

    except Exception as e:
        return {"user_query": user_query, "agent_used": "unknown", "response": f"Error: {str(e)}", "status": "error"}
