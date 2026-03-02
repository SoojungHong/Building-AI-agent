# Node 2: Execute Step
def execute_step(self, state: PlanExecuteState):
    """Execute the current step in the plan."""

    plan = state["plan_list"]
    current_step = state["current_step"]

    if current_step >= len(plan):
        return {"current_step": current_step}

    step = plan[current_step]
    step_id = step["step_id"]
    description = step["description"]
    depends_on = step.get("depends_on")
    tool_to_use = step.get("tool_to_use")

    # Get dependencies
    dependency_context = ""
    extracted_entities = {}  # NEW: store extracted IDs/values

    if depends_on:
        for dep_id in depends_on:
            dep_result = state["step_results"].get(f"step_{dep_id}")
            if dep_result:
                dependency_context += f"\nResult from step {dep_id}: {dep_result}"
                
                # NEW: Extract client_id from dependency result if present
                extracted_entities.update(
                    extract_entities_from_result(dep_result)
                )

    print("[DEBUG] Result from step : ", dependency_context)
    print("[DEBUG] Extracted entities : ", extracted_entities)  # NEW

    # Execute step with context
    system_prompt = f"""You are executing step {step_id} of a plan.

Step description: {description}

Previous results:
{dependency_context if dependency_context else "No dependencies"}

User ID: {state['user_id']}

# NEW: Pass extracted entities explicitly so tool uses correct IDs
Extracted entities from previous steps: {extracted_entities}

Execute this step and provide the result.
IMPORTANT: If there are previous step contains specific client ID or specific 
opportunity, YOU MUST use that exact client ID or opportunity ID when calling tools.
Do NOT search for a new client. Use the extracted entities above directly.

import re

def extract_entities_from_result(result: str) -> dict:
    """
    Extract structured IDs from a step result string.
    Adapt the patterns to match your actual result format.
    """
    entities = {}

    # Extract client ID (e.g. GFIW_UKR296767 or CLIENT_ID: xxx)
    client_id_match = re.search(
        r'CLIENT_ID[:\*\s]+([A-Z0-9_]+)', result, re.IGNORECASE
    )
    if client_id_match:
        entities["client_id"] = client_id_match.group(1)

    # Extract R-ID (e.g. R296767)
    rid_match = re.search(r'R-ID[:\*\s]+([A-Z0-9]+)', result, re.IGNORECASE)
    if rid_match:
        entities["r_id"] = rid_match.group(1)

    # Extract full name if present
    name_match = re.search(r'Full Name[:\*\s]+([^\n\*]+)', result, re.IGNORECASE)
    if name_match:
        entities["client_name"] = name_match.group(1).strip()

    return entities

