#Initialize an agent with agent_graph capability
from strands.multiagent import GraphBuilder
from strands import Agent
import os
from dotenv import load_dotenv
from strands import Agent
from strands.models.gemini import GeminiModel


load_dotenv()

model = GeminiModel(
    client_args={
        "api_key": os.getenv("GEMINI_API_KEY"),
    },
    # **model_config
    model_id="gemini-2.5-flash",
    params={
        # some sample model parameters 
        "temperature": 0.7,
        "max_output_tokens": 2048,
        "top_p": 0.9,
        "top_k": 40
    }
)

# Create specialized agents
coordinator = Agent(
    name="coordinator",
     system_prompt=""" You are a research team leader coordinating specialists.
      Provide a short analysis, no need for follow ups"""
      
      )
analyst = Agent(name="data_analyst", system_prompt="You are a data analyst specializing in statistical analysis. Provide a short analysis, no need for follow ups")
domain_expert = Agent(name="domain_expert", system_prompt="You are a domain expert with deep subject knowledge. Provide a short analysis, no need for follow ups")

# Build the graph
builder = GraphBuilder()

# Add nodes
builder.add_node(coordinator, "team_lead")
builder.add_node(analyst, "analyst")
builder.add_node(domain_expert, "expert")

# Add edges (dependencies)
builder.add_edge("team_lead", "analyst")
builder.add_edge("team_lead", "expert")

# Set entry points (optional - will be auto-detected if not specified)
builder.set_entry_point("team_lead")

# Build the graph
graph = builder.build()

#Execute task on newly built graph
result = graph("Analyze the impact of remote work on employee productivity.Provide a short analysis, no need for follow ups")
print("\n")