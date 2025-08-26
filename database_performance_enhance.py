from langchain.agents import create_sql_agent
from langchain.agents.agent_toolkits import SQLDatabaseToolkit
from langchain.sql_database import SQLDatabase
from langchain.llms import OpenAI
from langchain.agents import AgentExecutor
from langchain.prompts import PromptTemplate
from langchain.schema import SystemMessage
import sqlite3

# System instructions for SQL query optimization
SYSTEM_INSTRUCTIONS = """
You are an expert SQL query optimizer. Your goal is to generate the fastest possible SQL queries.

CRITICAL OPTIMIZATION RULES:
1. ALWAYS use appropriate indexes - prefer indexed columns in WHERE clauses
2. Use LIMIT clauses when possible to reduce result sets
3. Avoid SELECT * - only select needed columns
4. Use WHERE clauses to filter data as early as possible
5. Prefer EXISTS over IN for subqueries when checking existence
6. Use appropriate JOIN types (INNER JOIN is fastest when applicable)
7. Avoid functions in WHERE clauses that prevent index usage
8. Use UNION ALL instead of UNION when duplicates don't matter
9. Consider using WITH clauses for complex queries to improve readability and performance
10. Use appropriate aggregate functions and GROUP BY efficiently

QUERY STRUCTURE PRIORITIES:
- Filter first (WHERE), then join, then aggregate
- Use covering indexes when possible
- Avoid unnecessary sorting (ORDER BY) unless specifically requested
- Use EXPLAIN QUERY PLAN to verify optimization

PERFORMANCE GUIDELINES:
- For large tables, always include meaningful WHERE conditions
- Use parameterized queries to enable query plan caching
- Consider partitioning strategies for very large datasets
- Prefer set-based operations over row-by-row processing

When generating SQL:
1. Start with the most selective WHERE conditions
2. Join on indexed foreign keys
3. Only include necessary columns in SELECT
4. Add LIMIT when appropriate for the user's needs
5. Consider using subqueries vs JOINs based on data size and indexes
"""

class OptimizedSQLAgent:
    def __init__(self, database_uri: str, model_name: str = "gpt-3.5-turbo"):
        """
        Initialize the SQL agent with optimization focus
        
        Args:
            database_uri: Database connection string
            model_name: LLM model to use
        """
        # Initialize database connection
        self.db = SQLDatabase.from_uri(database_uri)
        
        # Initialize LLM with specific instructions
        self.llm = OpenAI(
            model_name=model_name,
            temperature=0,  # Deterministic for SQL generation
            max_tokens=2000
        )
        
        # Create SQL toolkit
        self.toolkit = SQLDatabaseToolkit(db=self.db, llm=self.llm)
        
        # Create optimized prompt template
        self.prompt_template = PromptTemplate(
            input_variables=["input", "agent_scratchpad", "table_info"],
            template="""
            {system_instructions}
            
            You have access to the following tables:
            {table_info}
            
            User Query: {input}
            
            Instructions:
            1. Analyze the user's request carefully
            2. Generate the most optimized SQL query possible
            3. Explain your optimization choices
            4. Use EXPLAIN QUERY PLAN if helpful for verification
            
            {agent_scratchpad}
            """
        )
        
        # Create the agent with optimization instructions
        self.agent = create_sql_agent(
            llm=self.llm,
            toolkit=self.toolkit,
            verbose=True,
            agent_kwargs={
                "system_message": SystemMessage(content=SYSTEM_INSTRUCTIONS),
                "extra_prompt_messages": [
                    SystemMessage(content="Focus on query performance and optimization.")
                ]
            }
        )
    
    def create_and_react(self, user_prompt: str) -> dict:
        """
        Process user prompt and return optimized SQL query with execution results
        
        Args:
            user_prompt: Natural language query from user
            
        Returns:
            Dictionary containing SQL query, results, and optimization notes
        """
        try:
            # Add optimization context to user prompt
            enhanced_prompt = f"""
            User request: {user_prompt}
            
            Please generate the most optimized SQL query for this request. 
            Consider performance implications and explain your optimization choices.
            If the query might be slow, suggest alternatives or additional indexes.
            """
            
            # Execute the agent
            result = self.agent.run(enhanced_prompt)
            
            return {
                "success": True,
                "result": result,
                "user_query": user_prompt,
                "optimization_applied": True
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "user_query": user_prompt
            }
    
    def analyze_query_plan(self, sql_query: str) -> str:
        """
        Analyze query execution plan for optimization insights
        
        Args:
            sql_query: SQL query to analyze
            
        Returns:
            Query plan analysis
        """
        try:
            explain_query = f"EXPLAIN QUERY PLAN {sql_query}"
            result = self.db.run(explain_query)
            return result
        except Exception as e:
            return f"Error analyzing query plan: {str(e)}"
    
    def suggest_indexes(self, table_name: str, columns: list) -> list:
        """
        Suggest indexes for better performance
        
        Args:
            table_name: Target table name
            columns: Columns frequently used in WHERE/JOIN clauses
            
        Returns:
            List of suggested CREATE INDEX statements
        """
        suggestions = []
        for col in columns:
            index_name = f"idx_{table_name}_{col}"
            index_sql = f"CREATE INDEX {index_name} ON {table_name} ({col});"
            suggestions.append(index_sql)
        
        return suggestions

# Example usage and testing
def create_sample_database():
    """Create a sample database for testing"""
    conn = sqlite3.connect('sample.db')
    cursor = conn.cursor()
    
    # Create sample tables
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS customers (
            id INTEGER PRIMARY KEY,
            name TEXT,
            email TEXT,
            city TEXT,
            created_date DATE
        )
    ''')
    
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS orders (
            id INTEGER PRIMARY KEY,
            customer_id INTEGER,
            product_name TEXT,
            amount DECIMAL(10,2),
            order_date DATE,
            FOREIGN KEY (customer_id) REFERENCES customers (id)
        )
    ''')
    
    # Create indexes for better performance
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_customers_city ON customers (city)')
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_orders_customer_id ON orders (customer_id)')
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_orders_date ON orders (order_date)')
    
    # Insert sample data
    customers_data = [
        (1, 'John Doe', 'john@email.com', 'New York', '2023-01-15'),
        (2, 'Jane Smith', 'jane@email.com', 'Los Angeles', '2023-02-20'),
        (3, 'Bob Johnson', 'bob@email.com', 'Chicago', '2023-03-10')
    ]
    
    orders_data = [
        (1, 1, 'Laptop', 1200.00, '2023-04-01'),
        (2, 1, 'Mouse', 25.00, '2023-04-02'),
        (3, 2, 'Keyboard', 75.00, '2023-04-03'),
        (4, 3, 'Monitor', 300.00, '2023-04-04')
    ]
    
    cursor.executemany('INSERT OR REPLACE INTO customers VALUES (?, ?, ?, ?, ?)', customers_data)
    cursor.executemany('INSERT OR REPLACE INTO orders VALUES (?, ?, ?, ?, ?)', orders_data)
    
    conn.commit()
    conn.close()

# Main execution example
def main():
    # Create sample database
    create_sample_database()
    
    # Initialize the optimized SQL agent
    agent = OptimizedSQLAgent("sqlite:///sample.db")
    
    # Example queries
    test_queries = [
        "Show me all customers from New York with their total order amounts",
        "Find the top 5 products by revenue in the last month",
        "List customers who haven't placed any orders",
        "Get the average order value by city"
    ]
    
    print("=== Optimized SQL Query Agent Demo ===\n")
    
    for i, query in enumerate(test_queries, 1):
        print(f"Query {i}: {query}")
        print("-" * 50)
        
        result = agent.create_and_react(query)
        
        if result["success"]:
            print("✅ Success!")
            print("Result:", result["result"])
        else:
            print("❌ Error:", result["error"])
        
        print("\n" + "="*70 + "\n")

if __name__ == "__main__":
    main()
