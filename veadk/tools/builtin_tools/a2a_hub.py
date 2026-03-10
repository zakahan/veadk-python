from google.adk.tools import ToolContext


def list_sub_agents(tool_context: ToolContext) -> list[dict] | str:
    """
    List all available A2A agents.

    Returns:
        list[dict]: A list of agent information, each element format:
            {
                "id": "agent id",
                "name": "agent name",
                "description": "agent description"
            }
        str: Error message (when a2a_space_id is not configured)
    """
    from veadk.utils.a2a_utils import list_remote_a2a_agents

    a2a_space_id = getattr(tool_context._invocation_context.agent, "a2a_space_id", None)
    a2a_space_config = getattr(
        tool_context._invocation_context.agent, "a2a_space_config", {}
    )
    if not a2a_space_id:
        return "Please configure the `a2a_space_id` in the agent."

    agents = list_remote_a2a_agents(
        a2a_space_id=a2a_space_id,
        a2a_space_config=a2a_space_config,
        output_mode="tool",
    )
    return agents


def add_sub_agents(agents: list[str], tool_context: ToolContext) -> str:
    """
    Create sub agents based on the provided agent id list and add them to the current agent's sub_agents.

    Args:
        agents: List of agent ids to create

    Returns:
        str: Operation result message
    """
    from veadk.utils.a2a_utils import get_a2a_agent, create_remote_ve_agent

    current_agent = tool_context._invocation_context.agent
    a2a_space_id = getattr(current_agent, "a2a_space_id", None)
    a2a_space_config = getattr(current_agent, "a2a_space_config", None)

    if not a2a_space_id:
        return "Please configure the `a2a_space_id` in the agent."

    created_agents = []
    failed_agents = []

    for agent_id in agents:
        try:
            agent_info = get_a2a_agent(
                space_id=a2a_space_id,
                agent_id=agent_id,
                version=a2a_space_config.get("version", "2025-10-30")
                if a2a_space_config
                else "2025-10-30",
            )

            if agent_info.get("Status", "NotFound").lower() != "running":
                failed_agents.append(f"{agent_id} (not running)")
                continue

            remote_agent = create_remote_ve_agent(agent_info)
            current_agent.sub_agents.append(remote_agent)
            created_agents.append(agent_id)
        except Exception as e:
            failed_agents.append(f"{agent_id} ({str(e)})")

    result_parts = []
    if created_agents:
        result_parts.append(f"Successfully created agents: {', '.join(created_agents)}")
    if failed_agents:
        result_parts.append(f"Failed to create agents: {', '.join(failed_agents)}")

    return "\n".join(result_parts) if result_parts else "No agents were created."
