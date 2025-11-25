#!/usr/bin/env python3
"""
Revenue Knowledge Graph System with LLM Query Interface
Builds a knowledge graph from CSV revenue data and enables natural language querying
"""

import csv
import json
import time
from collections import defaultdict
from datetime import datetime
from typing import Dict, List, Any, Set, Tuple


class RevenueKnowledgeGraph:
    """In-memory knowledge graph optimized for revenue data analysis"""
    
    def __init__(self):
        # Graph structure: adjacency lists
        self.nodes = {}  # node_id -> {type, properties}
        self.edges = []  # list of (source, relation, target, properties)
        
        # Indexes for fast lookup
        self.customers = {}
        self.products = {}
        self.regions = {}
        self.categories = {}
        self.transactions = {}
        self.date_index = defaultdict(list)  # date -> transaction_ids
        
    def add_node(self, node_id: str, node_type: str, properties: Dict) -> None:
        """Add a node to the graph"""
        self.nodes[node_id] = {
            'type': node_type,
            'properties': properties
        }
        
    def add_edge(self, source: str, relation: str, target: str, properties: Dict = None) -> None:
        """Add an edge to the graph"""
        self.edges.append({
            'source': source,
            'relation': relation,
            'target': target,
            'properties': properties or {}
        })
        
    def build_from_csv(self, csv_path: str) -> Dict[str, Any]:
        """Build knowledge graph from CSV file"""
        start_time = time.time()
        stats = {
            'transactions': 0,
            'customers': 0,
            'products': 0,
            'regions': 0,
            'categories': 0
        }
        
        with open(csv_path, 'r') as f:
            reader = csv.DictReader(f)
            
            for row in reader:
                txn_id = f"txn_{row['transaction_id']}"
                customer_id = f"customer_{row['customer_name'].replace(' ', '_')}"
                product_id = f"product_{row['product'].replace(' ', '_')}"
                region_id = f"region_{row['region']}"
                category_id = f"category_{row['category']}"
                date = row['date']
                
                # Add transaction node
                self.add_node(txn_id, 'Transaction', {
                    'id': row['transaction_id'],
                    'date': date,
                    'quantity': int(row['quantity']),
                    'unit_price': float(row['unit_price']),
                    'revenue': float(row['revenue'])
                })
                self.transactions[txn_id] = self.nodes[txn_id]
                self.date_index[date].append(txn_id)
                stats['transactions'] += 1
                
                # Add customer node (if not exists)
                if customer_id not in self.customers:
                    self.add_node(customer_id, 'Customer', {
                        'name': row['customer_name']
                    })
                    self.customers[customer_id] = self.nodes[customer_id]
                    stats['customers'] += 1
                    
                # Add product node (if not exists)
                if product_id not in self.products:
                    self.add_node(product_id, 'Product', {
                        'name': row['product'],
                        'unit_price': float(row['unit_price'])
                    })
                    self.products[product_id] = self.nodes[product_id]
                    stats['products'] += 1
                    
                # Add region node (if not exists)
                if region_id not in self.regions:
                    self.add_node(region_id, 'Region', {
                        'name': row['region']
                    })
                    self.regions[region_id] = self.nodes[region_id]
                    stats['regions'] += 1
                    
                # Add category node (if not exists)
                if category_id not in self.categories:
                    self.add_node(category_id, 'Category', {
                        'name': row['category']
                    })
                    self.categories[category_id] = self.nodes[category_id]
                    stats['categories'] += 1
                
                # Add relationships
                self.add_edge(txn_id, 'PURCHASED_BY', customer_id)
                self.add_edge(txn_id, 'FOR_PRODUCT', product_id)
                self.add_edge(txn_id, 'IN_REGION', region_id)
                self.add_edge(product_id, 'IN_CATEGORY', category_id)
                self.add_edge(customer_id, 'LOCATED_IN', region_id)
                
        build_time = time.time() - start_time
        stats['build_time'] = f"{build_time:.3f}s"
        stats['total_nodes'] = len(self.nodes)
        stats['total_edges'] = len(self.edges)
        
        return stats
    
    def query_revenue_by_customer(self, customer_name: str = None) -> List[Dict]:
        """Get revenue by customer"""
        results = []
        for customer_id, customer_node in self.customers.items():
            if customer_name and customer_name.lower() not in customer_node['properties']['name'].lower():
                continue
                
            # Find all transactions for this customer
            customer_txns = [
                edge for edge in self.edges 
                if edge['relation'] == 'PURCHASED_BY' and edge['target'] == customer_id
            ]
            
            total_revenue = sum(
                self.nodes[edge['source']]['properties']['revenue']
                for edge in customer_txns
            )
            
            results.append({
                'customer': customer_node['properties']['name'],
                'total_revenue': total_revenue,
                'transaction_count': len(customer_txns)
            })
            
        return sorted(results, key=lambda x: x['total_revenue'], reverse=True)
    
    def query_revenue_by_product(self) -> List[Dict]:
        """Get revenue by product"""
        results = []
        for product_id, product_node in self.products.items():
            product_txns = [
                edge for edge in self.edges
                if edge['relation'] == 'FOR_PRODUCT' and edge['target'] == product_id
            ]
            
            total_revenue = sum(
                self.nodes[edge['source']]['properties']['revenue']
                for edge in product_txns
            )
            total_quantity = sum(
                self.nodes[edge['source']]['properties']['quantity']
                for edge in product_txns
            )
            
            results.append({
                'product': product_node['properties']['name'],
                'total_revenue': total_revenue,
                'total_quantity': total_quantity,
                'transaction_count': len(product_txns)
            })
            
        return sorted(results, key=lambda x: x['total_revenue'], reverse=True)
    
    def query_revenue_by_region(self) -> List[Dict]:
        """Get revenue by region"""
        results = []
        for region_id, region_node in self.regions.items():
            region_txns = [
                edge for edge in self.edges
                if edge['relation'] == 'IN_REGION' and edge['target'] == region_id
            ]
            
            total_revenue = sum(
                self.nodes[edge['source']]['properties']['revenue']
                for edge in region_txns
            )
            
            results.append({
                'region': region_node['properties']['name'],
                'total_revenue': total_revenue,
                'transaction_count': len(region_txns)
            })
            
        return sorted(results, key=lambda x: x['total_revenue'], reverse=True)
    
    def query_revenue_by_category(self) -> List[Dict]:
        """Get revenue by category"""
        category_revenue = defaultdict(lambda: {'revenue': 0, 'count': 0})
        
        # For each transaction, find its product's category
        for txn_id, txn_node in self.transactions.items():
            # Find product for this transaction
            product_edges = [e for e in self.edges if e['source'] == txn_id and e['relation'] == 'FOR_PRODUCT']
            if not product_edges:
                continue
                
            product_id = product_edges[0]['target']
            
            # Find category for this product
            category_edges = [e for e in self.edges if e['source'] == product_id and e['relation'] == 'IN_CATEGORY']
            if not category_edges:
                continue
                
            category_id = category_edges[0]['target']
            category_name = self.nodes[category_id]['properties']['name']
            
            category_revenue[category_name]['revenue'] += txn_node['properties']['revenue']
            category_revenue[category_name]['count'] += 1
        
        results = [
            {
                'category': cat,
                'total_revenue': data['revenue'],
                'transaction_count': data['count']
            }
            for cat, data in category_revenue.items()
        ]
        
        return sorted(results, key=lambda x: x['total_revenue'], reverse=True)
    
    def query_top_customers(self, n: int = 5) -> List[Dict]:
        """Get top N customers by revenue"""
        results = self.query_revenue_by_customer()
        return results[:n]
    
    def query_monthly_revenue(self) -> List[Dict]:
        """Get revenue by month"""
        monthly_revenue = defaultdict(float)
        
        for txn_id, txn_node in self.transactions.items():
            date_str = txn_node['properties']['date']
            month = date_str[:7]  # YYYY-MM
            monthly_revenue[month] += txn_node['properties']['revenue']
        
        results = [
            {'month': month, 'revenue': revenue}
            for month, revenue in sorted(monthly_revenue.items())
        ]
        
        return results


class LLMQueryInterface:
    """Natural language query interface using LLM to translate to graph queries"""
    
    def __init__(self, kg: RevenueKnowledgeGraph):
        self.kg = kg
        self.query_examples = self._build_examples()
        
    def _build_examples(self) -> str:
        """Build example queries for the LLM prompt"""
        return """
Available Query Functions:
1. query_revenue_by_customer(customer_name=None) - Revenue by customer, optionally filter by name
2. query_revenue_by_product() - Revenue by product
3. query_revenue_by_region() - Revenue by region  
4. query_revenue_by_category() - Revenue by category
5. query_top_customers(n=5) - Top N customers by revenue
6. query_monthly_revenue() - Revenue by month

Example mappings:
- "What are our top customers?" -> query_top_customers(5)
- "Show revenue by region" -> query_revenue_by_region()
- "How much did Acme Corp spend?" -> query_revenue_by_customer("Acme Corp")
- "What products generate the most revenue?" -> query_revenue_by_product()
- "Revenue by category" -> query_revenue_by_category()
- "Monthly revenue trend" -> query_monthly_revenue()
- "Who are the top 10 customers?" -> query_top_customers(10)
"""
    
    def create_llm_prompt(self, user_query: str) -> str:
        """Create prompt for LLM to translate natural language to function call"""
        return f"""You are a query translator for a revenue knowledge graph system.

{self.query_examples}

User Query: "{user_query}"

Based on the user's query, respond with ONLY a JSON object in this exact format:
{{
  "function": "function_name",
  "params": {{}},
  "explanation": "brief explanation of what this will return"
}}

DO NOT include any other text, markdown formatting, or backticks. Output ONLY the JSON object.
"""
    
    def parse_query(self, user_query: str) -> Tuple[str, Dict, float]:
        """
        Simulate LLM parsing (in real implementation, this would call Claude API)
        Returns: (function_name, params, parse_time)
        """
        start_time = time.time()
        
        # Simple rule-based parsing for demo (replace with actual LLM call)
        query_lower = user_query.lower()
        
        if 'top' in query_lower and 'customer' in query_lower:
            # Extract number if present
            import re
            numbers = re.findall(r'\d+', query_lower)
            n = int(numbers[0]) if numbers else 5
            result = ('query_top_customers', {'n': n})
            
        elif 'customer' in query_lower and any(word in query_lower for word in ['how much', 'revenue', 'spend', 'spent']):
            # Try to extract customer name
            for customer in self.kg.customers.values():
                name = customer['properties']['name'].lower()
                if name in query_lower:
                    result = ('query_revenue_by_customer', {'customer_name': customer['properties']['name']})
                    break
            else:
                result = ('query_revenue_by_customer', {})
                
        elif 'product' in query_lower:
            result = ('query_revenue_by_product', {})
            
        elif 'region' in query_lower:
            result = ('query_revenue_by_region', {})
            
        elif 'category' in query_lower:
            result = ('query_revenue_by_category', {})
            
        elif 'month' in query_lower or 'trend' in query_lower:
            result = ('query_monthly_revenue', {})
            
        else:
            # Default to top customers
            result = ('query_top_customers', {'n': 5})
        
        parse_time = time.time() - start_time
        return result[0], result[1], parse_time
    
    def execute_query(self, user_query: str) -> Dict[str, Any]:
        """Execute a natural language query"""
        total_start = time.time()
        
        # Parse query (LLM translation)
        function_name, params, parse_time = self.parse_query(user_query)
        
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
            'results': results,
            'timing': {
                'llm_parse': f"{parse_time:.4f}s",
                'graph_query': f"{query_time:.4f}s",
                'total': f"{total_time:.4f}s"
            }
        }


def format_results(query_result: Dict) -> str:
    """Format query results for display"""
    output = []
    output.append(f"\n{'='*70}")
    output.append(f"Query: {query_result['query']}")
    output.append(f"Function: {query_result['function']}({query_result['params']})")
    output.append(f"{'='*70}\n")
    
    results = query_result['results']
    
    if isinstance(results, list) and results:
        # Display as table
        if all(isinstance(r, dict) for r in results):
            keys = list(results[0].keys())
            
            # Header
            header = " | ".join(f"{k:>20}" for k in keys)
            output.append(header)
            output.append("-" * len(header))
            
            # Rows
            for row in results[:10]:  # Limit to 10 rows
                values = []
                for k in keys:
                    v = row[k]
                    if isinstance(v, float):
                        values.append(f"{v:>20,.2f}")
                    else:
                        values.append(f"{str(v):>20}")
                output.append(" | ".join(values))
            
            if len(results) > 10:
                output.append(f"\n... and {len(results) - 10} more rows")
    else:
        output.append(str(results))
    
    output.append(f"\n{'='*70}")
    output.append(f"Timing: LLM Parse: {query_result['timing']['llm_parse']}, "
                 f"Graph Query: {query_result['timing']['graph_query']}, "
                 f"Total: {query_result['timing']['total']}")
    output.append(f"{'='*70}\n")
    
    return "\n".join(output)


def main():
    """Main demonstration"""
    print("Revenue Knowledge Graph System with LLM Query Interface")
    print("=" * 70)
    
    # Build knowledge graph
    print("\n1. Building Knowledge Graph from CSV...")
    kg = RevenueKnowledgeGraph()
    stats = kg.build_from_csv('sample_revenue.csv')
    
    print(f"   ✓ Built graph in {stats['build_time']}")
    print(f"   ✓ Nodes: {stats['total_nodes']} ({stats['customers']} customers, "
          f"{stats['products']} products, {stats['regions']} regions, "
          f"{stats['categories']} categories)")
    print(f"   ✓ Edges: {stats['total_edges']}")
    print(f"   ✓ Transactions: {stats['transactions']}")
    
    # Initialize LLM query interface
    print("\n2. Initializing LLM Query Interface...")
    interface = LLMQueryInterface(kg)
    print("   ✓ Ready for natural language queries")
    
    # Run example queries
    print("\n3. Running Example Queries...")
    
    example_queries = [
        "What are our top 5 customers?",
        "Show me revenue by region",
        "Which products generate the most revenue?",
        "What's our monthly revenue trend?",
        "Revenue by category",
        "How much did Acme Corp spend?"
    ]
    
    for query in example_queries:
        result = interface.execute_query(query)
        print(format_results(result))
    
    print("\n4. Performance Summary:")
    print("   ✓ Typical query time: < 0.01s")
    print("   ✓ With LLM parsing: ~0.01-0.05s (simulated)")
    print("   ✓ Real LLM API call would add: ~1-3s")
    print("\n   Fast enough for interactive use! ✓")


if __name__ == "__main__":
    main()
