#!/usr/bin/env python3
"""
Revenue Knowledge Graph with Real LLM Integration
Uses Claude API for natural language query parsing
"""

import json
import time
import requests
from revenue_kg_system import RevenueKnowledgeGraph, format_results


class ClaudeLLMQueryInterface:
    """Natural language query interface using Claude API"""
    
    def __init__(self, kg: RevenueKnowledgeGraph):
        self.kg = kg
        self.api_url = "https://api.anthropic.com/v1/messages"
        
    def create_llm_prompt(self, user_query: str) -> str:
        """Create prompt for Claude to translate natural language to function call"""
        
        available_functions = """
Available Query Functions:
1. query_revenue_by_customer(customer_name=None) - Revenue by customer. If customer_name is provided, filters to that customer.
2. query_revenue_by_product() - Revenue breakdown by product
3. query_revenue_by_region() - Revenue breakdown by region  
4. query_revenue_by_category() - Revenue breakdown by category
5. query_top_customers(n=5) - Top N customers by revenue (default n=5)
6. query_monthly_revenue() - Revenue aggregated by month

Examples:
- "What are our top customers?" → query_top_customers(5)
- "Show revenue by region" → query_revenue_by_region()
- "How much did Acme Corp spend?" → query_revenue_by_customer("Acme Corp")
- "What products make the most money?" → query_revenue_by_product()
- "Revenue by category" → query_revenue_by_category()
- "Monthly revenue trend" → query_monthly_revenue()
- "Who are the top 10 customers?" → query_top_customers(10)
"""
        
        return f"""{available_functions}

User Query: "{user_query}"

Analyze the user's query and determine which function to call with what parameters.

Respond with ONLY a valid JSON object in this format (no markdown, no backticks, no other text):
{{
  "function": "function_name",
  "params": {{}},
  "explanation": "what this query will return"
}}

DO NOT OUTPUT ANYTHING OTHER THAN VALID JSON."""
    
    def parse_query_with_llm(self, user_query: str) -> tuple:
        """Use Claude API to parse natural language query"""
        start_time = time.time()
        
        try:
            prompt = self.create_llm_prompt(user_query)
            
            response = requests.post(
                self.api_url,
                headers={"Content-Type": "application/json"},
                json={
                    "model": "claude-sonnet-4-20250514",
                    "max_tokens": 500,
                    "messages": [
                        {"role": "user", "content": prompt}
                    ]
                }
            )
            
            if response.status_code == 200:
                data = response.json()
                response_text = data['content'][0]['text'].strip()
                
                # Clean up response (remove markdown formatting if present)
                response_text = response_text.replace('```json\n', '').replace('```\n', '').replace('```', '').strip()
                
                # Parse JSON response
                parsed = json.loads(response_text)
                
                parse_time = time.time() - start_time
                return (
                    parsed['function'],
                    parsed.get('params', {}),
                    parse_time,
                    parsed.get('explanation', '')
                )
            else:
                print(f"API Error: {response.status_code}")
                print(f"Response: {response.text}")
                # Fallback to simple parsing
                return self._simple_parse(user_query)
                
        except Exception as e:
            print(f"LLM parsing error: {e}")
            # Fallback to simple rule-based parsing
            return self._simple_parse(user_query)
    
    def _simple_parse(self, user_query: str) -> tuple:
        """Fallback simple rule-based parsing"""
        start_time = time.time()
        query_lower = user_query.lower()
        
        if 'top' in query_lower and 'customer' in query_lower:
            import re
            numbers = re.findall(r'\d+', query_lower)
            n = int(numbers[0]) if numbers else 5
            result = ('query_top_customers', {'n': n}, time.time() - start_time, 'Top customers by revenue')
        elif 'product' in query_lower:
            result = ('query_revenue_by_product', {}, time.time() - start_time, 'Revenue by product')
        elif 'region' in query_lower:
            result = ('query_revenue_by_region', {}, time.time() - start_time, 'Revenue by region')
        elif 'category' in query_lower:
            result = ('query_revenue_by_category', {}, time.time() - start_time, 'Revenue by category')
        elif 'month' in query_lower or 'trend' in query_lower:
            result = ('query_monthly_revenue', {}, time.time() - start_time, 'Monthly revenue trend')
        else:
            result = ('query_top_customers', {'n': 5}, time.time() - start_time, 'Top customers')
            
        return result
    
    def execute_query(self, user_query: str, use_llm: bool = True) -> dict:
        """Execute a natural language query"""
        total_start = time.time()
        
        # Parse query
        if use_llm:
            function_name, params, parse_time, explanation = self.parse_query_with_llm(user_query)
        else:
            function_name, params, parse_time, explanation = self._simple_parse(user_query)
        
        # Execute graph query
        query_start = time.time()
        function = getattr(self.kg, function_name)
        results = function(**params)
        query_time = time.time() - query_start
        
        total_time = time.time() - total_start
        
        return {
            'query': user_query,
            'function': function_name,
            'params': params,
            'explanation': explanation,
            'results': results,
            'timing': {
                'llm_parse': f"{parse_time:.4f}s",
                'graph_query': f"{query_time:.4f}s",
                'total': f"{total_time:.4f}s"
            }
        }


def demo_with_llm():
    """Demo with actual LLM integration"""
    print("Revenue Knowledge Graph with Claude LLM Integration")
    print("=" * 70)
    
    # Build knowledge graph
    print("\n1. Building Knowledge Graph...")
    kg = RevenueKnowledgeGraph()
    stats = kg.build_from_csv('sample_revenue.csv')
    print(f"   ✓ Graph built in {stats['build_time']}")
    print(f"   ✓ {stats['total_nodes']} nodes, {stats['total_edges']} edges")
    
    # Initialize LLM interface
    print("\n2. Initializing Claude LLM Interface...")
    interface = ClaudeLLMQueryInterface(kg)
    print("   ✓ Ready for natural language queries with Claude API")
    
    # Test queries
    print("\n3. Testing Natural Language Queries with Claude...")
    
    test_queries = [
        "What are our top 3 customers by revenue?",
        "Show me which region generates the most sales",
        "I want to see product performance",
    ]
    
    for query in test_queries:
        print(f"\n   Testing: '{query}'")
        result = interface.execute_query(query, use_llm=True)
        print(f"   → Function: {result['function']}({result['params']})")
        print(f"   → Parse time: {result['timing']['llm_parse']}")
        print(f"   → Query time: {result['timing']['graph_query']}")
        print(f"   → Total time: {result['timing']['total']}")


def interactive_mode():
    """Interactive query mode"""
    print("Revenue Knowledge Graph - Interactive Mode")
    print("=" * 70)
    
    # Build graph
    kg = RevenueKnowledgeGraph()
    stats = kg.build_from_csv('sample_revenue.csv')
    print(f"Loaded {stats['transactions']} transactions")
    
    # Initialize interface
    interface = ClaudeLLMQueryInterface(kg)
    
    print("\nYou can ask questions like:")
    print("  - What are our top customers?")
    print("  - Show revenue by region")
    print("  - Which products perform best?")
    print("  - What's the monthly trend?")
    print("\nType 'quit' to exit\n")
    
    while True:
        try:
            user_input = input("Your query: ").strip()
            
            if user_input.lower() in ['quit', 'exit', 'q']:
                break
            
            if not user_input:
                continue
            
            # Execute query (using simple parsing for demo, set use_llm=True for API)
            result = interface.execute_query(user_input, use_llm=False)
            print(format_results(result))
            
        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"Error: {e}")
    
    print("\nGoodbye!")


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == 'interactive':
        interactive_mode()
    else:
        demo_with_llm()
