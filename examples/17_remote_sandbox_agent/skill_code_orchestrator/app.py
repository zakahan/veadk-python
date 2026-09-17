"""Local and AgentKit Runtime entrypoint."""

from pathlib import Path

from dotenv import load_dotenv


def main():
    load_dotenv(Path(__file__).with_name(".env"))
    from agent import root_agent
    from agentkit.apps import AgentkitAgentServerApp

    AgentkitAgentServerApp(agent=root_agent).run(host="0.0.0.0", port=8000)


if __name__ == "__main__":
    main()
