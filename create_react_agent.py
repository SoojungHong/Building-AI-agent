# ----------------------------------------------------
# Option 1: Use Lambda Functions
# The lambda approach (Option 1) is the most straightforward when you have existing class methods you want to use as tools. Just make sure the description clearly states what input format the tool expects!
#-----------------------------------------------------

class MyAgent:
    def __init__(self):
        self.data = "some data"
        
    def search_tool(self, query: str) -> str:
        """Search using class data"""
        return f"Searching with {self.data}: {query}"
    
    def another_tool(self, query: str) -> str:
        """Another tool"""
        return f"Processing: {query}"
    
    def create_agent(self):
        from langchain.tools import Tool
        from langchain.agents import create_react_agent, AgentExecutor
        from langchain_openai import ChatOpenAI
        
        # Wrap self methods in lambda
        tools = [
            Tool(
                name="SearchTool",
                func=lambda q: self.search_tool(q),
                description="Useful for searching. Input: search query string"
            ),
            Tool(
                name="AnotherTool",
                func=lambda q: self.another_tool(q),
                description="Processes queries. Input: query string"
            )
        ]
        
        llm = ChatOpenAI(temperature=0)
        agent = create_react_agent(llm, tools, prompt)
        agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)
        
        return agent_executor


#---------------------------------------------
# Option 2: Use StructuredTool.from_function
#---------------------------------------------

from langchain.tools import StructuredTool

class MyAgent:
    def search_tool(self, query: str) -> str:
        """Search using class data"""
        return f"Result: {query}"
    
    def create_agent(self):
        tools = [
            StructuredTool.from_function(
                func=self.search_tool,
                name="SearchTool",
                description="Search tool. Input: query string"
            )
        ]
        
        # Create agent as usual
        agent = create_react_agent(llm, tools, prompt)
        return AgentExecutor(agent=agent, tools=tools)


#--------------------------------------------
# Option 3: Use @tool Decorator with Self
#--------------------------------------------

from langchain.tools import tool

class MyAgent:
    def __init__(self):
        self.database = {}
    
    def create_agent(self):
        # Create tool inside the method where self is available
        @tool
        def search_database(query: str) -> str:
            """Search the database. Input: search query"""
            return f"Searching {self.database} for: {query}"
        
        tools = [search_database]
        
        agent = create_react_agent(llm, tools, prompt)
        return AgentExecutor(agent=agent, tools=tools, verbose=True)


#---------------------------
# Basic Tool setup
#---------------------------
from langchain.agents import create_react_agent, AgentExecutor
from langchain.tools import Tool
from langchain_openai import ChatOpenAI

# Define your tool function
def my_search_tool(query: str) -> str:
    """Search for information based on the query."""
    # Your logic here
    result = f"Searching for: {query}"
    return result

# Create the tool
tools = [
    Tool(
        name="SearchTool",
        func=my_search_tool,
        description="Useful for searching information. Input should be a search query string."
    )
]

# Create agent
llm = ChatOpenAI(temperature=0)
agent = create_react_agent(llm, tools, prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

# Run - the agent will pass the query to your tool
result = agent_executor.invoke({"input": "What is the weather?"})
  
