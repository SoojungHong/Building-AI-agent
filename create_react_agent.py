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


#-------------------
#   improvement 
#-------------------
from langchain.tools import StructuredTool
from langchain.agents import create_react_agent, AgentExecutor
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate

class MyAgent:
    def __init__(self):
        self.database = {"products": ["laptop", "phone"], "prices": [1000, 500]}
        self.step_results = []  # Store results manually if needed
        
    def search_database(self, query: str) -> str:
        """Search the product database.
        
        Args:
            query: Product name or category to search for
            
        Returns:
            Matching products with prices
        """
        result = f"Found in DB: {self.database} matching '{query}'"
        self.step_results.append(("search_database", query, result))
        return result
    
    def calculate_price(self, product: str) -> str:
        """Calculate total price with tax.
        
        Args:
            product: Product name to calculate price for
            
        Returns:
            Price with tax calculation
        """
        result = f"Price calculation for {product}: $1000 + tax"
        self.step_results.append(("calculate_price", product, result))
        return result
    
    def create_agent(self):
        # Create tools with detailed descriptions
        tools = [
            StructuredTool.from_function(
                func=self.search_database,
                name="search_database",
                description="""Search the product database for items.
                Use when: User asks what products exist, product features, or availability.
                Input: Product name or category (string).
                Example input: 'laptop' or 'smartphones under $500'
                Returns: List of matching products."""
            ),
            StructuredTool.from_function(
                func=self.calculate_price,
                name="calculate_price",
                description="""Calculate final price including tax for a product.
                Use when: User asks about total cost, final price, or price with tax.
                Input: Exact product name (string).
                Example input: 'laptop X200'
                Returns: Price breakdown with tax."""
            )
        ]
        
        # Better prompt
        prompt = PromptTemplate.from_template("""Answer the following question using these tools.

Available tools:
{tools}

Tool names: {tool_names}

Use this format:
Question: the input question
Thought: think about what to do
Action: the tool to use (must be one of [{tool_names}])
Action Input: the input for the tool
Observation: the result from the tool
... (repeat Thought/Action/Action Input/Observation as needed)
Thought: I now know the final answer
Final Answer: the final answer to the question

Question: {input}
Thought: {agent_scratchpad}""")
        
        llm = ChatOpenAI(temperature=0, model="gpt-4")
        agent = create_react_agent(llm, tools, prompt)
        
        # KEY: Enable intermediate steps capture
        agent_executor = AgentExecutor(
            agent=agent,
            tools=tools,
            verbose=True,
            return_intermediate_steps=True,  # CRITICAL
            max_iterations=15,
            early_stopping_method="generate",
            handle_parsing_errors=True  # Graceful error handling
        )
        
        return agent_executor
    
    def run_query(self, query: str):
        """Run query and store all results"""
        self.step_results = []  # Reset
        agent_executor = self.create_agent()
        
        result = agent_executor.invoke({"input": query})
        
        # Store intermediate steps
        all_steps = result.get("intermediate_steps", [])
        
        return {
            "final_answer": result["output"],
            "all_steps": all_steps,
            "manual_log": self.step_results  # Your manual tracking
        }

# Usage
agent = MyAgent()
result = agent.run_query("What laptops do you have and what's the total price with tax?")

print("Final Answer:", result["final_answer"])
print("\nAll Steps:")
for action, observation in result["all_steps"]:
    print(f"\n  Tool: {action.tool}")
    print(f"  Input: {action.tool_input}")
    print(f"  Output: {observation}")
  
