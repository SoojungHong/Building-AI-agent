import concurrent.futures


def _run_single_step(self, step, dependency_snapshot, state):
    """Execute exactly one plan step and return its output plus metadata.

    This is the same logic that used to live inline inside execute_step(),
    just parameterized so it can be called for any step, from any thread.
    """
    step_id = step["step_id"]
    description = step["description"]
    depends_on = step.get("depends_on")
    tool_to_use = step.get("tool_to_use")

    # Build dependency context from the snapshot taken at the start of this round
    dependency_context = ""
    if depends_on:
        for dep_id in depends_on:
            dep_result = dependency_snapshot.get(f"step_{dep_id}")
            if dep_result:
                dependency_context += f"\nResult from step {dep_id}: {dep_result}\n"

    system_prompt = f"""You are executing step {step_id} of a plan.

Step description: {description}

Previous results:
{dependency_context if dependency_context else "No dependencies"}

User ID: {state['user_id']}

IMPORTANT: You MUST use the tool named '{tool_to_use}' for this step. Do not use any other tool.

If the step description includes "this client", "that client", "the client", etc, then replace the phrase 'this client' or 'that client'
If the step description includes "my", then utilize the user profile that is identified with User ID which is user email.
The result of the step should have summarized and concise form, therefore the result of step should not have more than 1000 characters.
Please complete the following task and return the output.
"""

    step_tool = next((t for t in self.test_tools if t.name == tool_to_use), None)

    if step_tool:
        print("\t\t[debug] step_tool : ", str(step_tool))
        # Force the LLM to use the specific tool
        forced_llm = self.azure_chat_client.bind_tools(
            self.test_tools,
            tool_choice=tool_to_use  # forces exact tool
        )

        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=f"Task: {description}")
        ]

        response = forced_llm.invoke(messages)

        # Extract tool call and execute the CORRECT tool directly
        if response.tool_calls:
            tool_call = response.tool_calls[0]

            # Force use of the planned tool, ignore what LLM picked
            tool_to_execute = next(
                (t for t in self.test_tools if t.name == tool_to_use),
                None
            )

            if tool_to_execute:
                tool_result = tool_to_execute.invoke(tool_call["args"])
                response = tool_result
    else:
        response = self.agent_executor.invoke(
            {"messages": [
                ("system", system_prompt),
                ("human", description)
            ]},
            config={"recursion_limit": 25}
        )

    # Normalize the response into (all_steps, output)
    if isinstance(response, dict):
        all_steps = response.get("messages", [])
        last_message = response["messages"][-1]
        output = last_message.content if hasattr(last_message, 'content') else str(response)
    else:
        # response is an AIMessage / tool result
        all_steps = [response]
        output = response.content if hasattr(response, 'content') else str(response)

    client_identifier = self.detect_client_id_in_prompt(str(output))

    return {
        "step_id": step_id,
        "description": description,
        "output": output,
        "all_steps": all_steps,
        "client_identifier": client_identifier,
    }


def execute_step(self, state: PlanExecuteState):
    """Execute every plan step whose dependencies are already satisfied.

    Steps with no unmet dependencies run concurrently (thread pool) instead
    of one-at-a-time. `current_step` is now a count of completed steps
    (used only to know when the whole plan is done), not an index.
    """
    plan = state["plan_list"]
    step_results = dict(state.get("step_results", {}))
    state['user_id'] = self.current_gpn_email

    completed_ids = {
        s["step_id"] for s in plan
        if f"step_{s['step_id']}" in step_results
    }

    # Steps that haven't run yet and whose dependencies are all completed
    ready_steps = [
        s for s in plan
        if s["step_id"] not in completed_ids
        and all(dep in completed_ids for dep in (s.get("depends_on") or []))
    ]

    if not ready_steps:
        # Nothing left to run (either done, or a dependency cycle — see note below)
        return {"current_step": len(plan)}

    print(f"\t\t[debug] running {len(ready_steps)} step(s) in parallel: "
          f"{[s['step_id'] for s in ready_steps]}")

    combined_all_steps = []
    combined_messages = []

    with concurrent.futures.ThreadPoolExecutor(max_workers=len(ready_steps)) as executor:
        futures = {
            executor.submit(self._run_single_step, step, step_results, state): step
            for step in ready_steps
        }
        for future in concurrent.futures.as_completed(futures):
            step = futures[future]
            try:
                result = future.result()
            except Exception as exc:
                print(f"\t\t[error] step {step['step_id']} failed: {exc}")
                result = {
                    "step_id": step["step_id"],
                    "description": step["description"],
                    "output": f"ERROR: {exc}",
                    "all_steps": [],
                    "client_identifier": "",
                }

            step_id = result["step_id"]
            output = result["output"]

            step_results[f"step_{step_id}"] = output
            self.step_results[f"step_{step_id}"] = (
                "The question is " + result["description"] +
                " and the response is " + str(output)
            )

            if self.current_gpn_email not in self.CLIENT_DATABASE:
                self.CLIENT_DATABASE[self.current_gpn_email] = {"client_identifier": ""}
            if result["client_identifier"]:
                self.CLIENT_DATABASE[self.current_gpn_email]["client_identifier"] = result["client_identifier"]

            combined_all_steps.extend(result["all_steps"])
            combined_messages.append(
                HumanMessage(content=f"Step {step_id} completed: {output}...")
            )

    return {
        "all_steps": combined_all_steps,
        "manual_log": self.step_results,
        "step_results": step_results,
        "current_step": state.get("current_step", 0) + len(ready_steps),
        "messages": combined_messages,
    }
