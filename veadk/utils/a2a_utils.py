import json
import os
from typing import Optional

from veadk.config import getenv
from veadk.utils.volcengine_sign import ve_request
from veadk.auth.veauth.utils import get_credential_from_vefaas_iam
from veadk.utils.logger import logger


def list_a2a_agents(
    space_id: str,
    project_name: str = "default",
    page_number: int = 1,
    page_size: int = 10,
    version: str = "2025-10-30",
):
    service = getenv("AGENTKIT_A2A_SERVICE_CODE", "agentkit")

    cloud_provider = (os.getenv("CLOUD_PROVIDER") or "").lower()
    if cloud_provider == "byteplus":
        raise ValueError("A2A is not supported for byteplus provider now.")
    else:
        sld = "volcengineapi"
        default_region = "cn-beijing"

    region = getenv("AGENTKIT_A2A_REGION", default_region)
    host = getenv("AGENTKIT_A2A_HOST", service + "." + region + f".{sld}.com")
    scheme = getenv("AGENTKIT_A2A_SCHEME", "https", allow_false_values=True).lower()
    if scheme not in {"http", "https"}:
        scheme = "https"

    ak = os.getenv("VOLCENGINE_ACCESS_KEY")
    sk = os.getenv("VOLCENGINE_SECRET_KEY")
    header = {}

    if not (ak and sk):
        logger.debug(
            "Get AK/SK from environment variables failed. Try to use credential from Iam."
        )
        credential = get_credential_from_vefaas_iam()
        ak = credential.access_key_id
        sk = credential.secret_access_key
        header = {"X-Security-Token": credential.session_token}
    else:
        logger.debug("Successfully get AK/SK from environment variables.")

    result = ve_request(
        request_body={
            "SpaceId": space_id,
            "ProjectName": project_name,
            "PageNumber": page_number,
            "PageSize": page_size,
        },
        action="ListA2aAgents",
        ak=ak,
        sk=sk,
        service=service,
        version=version,
        region=region,
        host=host,
        header=header,
        scheme=scheme,  # noqa
    )
    response_metadata = result.get("ResponseMetadata", {})
    if "Error" in response_metadata:
        logger.error(f"Agentkit A2A RequestMetadata: {response_metadata}.")
    else:
        logger.debug(f"Agentkit A2A RequestMetadata: {response_metadata}.")
    agent_list = result.get("Result", {}).get("Items", [])
    return agent_list


def get_a2a_agent(
    space_id: str,
    agent_id: Optional[str] = None,
    name: Optional[str] = None,
    runtime_id: Optional[str] = None,
    version: str = "2025-10-30",
):
    service = getenv("AGENTKIT_A2A_SERVICE_CODE", "agentkit")

    cloud_provider = (os.getenv("CLOUD_PROVIDER") or "").lower()
    if cloud_provider == "byteplus":
        raise ValueError("A2A is not supported for byteplus provider now.")
    else:
        sld = "volcengineapi"
        default_region = "cn-beijing"

    region = getenv("AGENTKIT_A2A_REGION", default_region)
    host = getenv("AGENTKIT_A2A_HOST", service + "." + region + f".{sld}.com")
    scheme = getenv("AGENTKIT_A2A_SCHEME", "https", allow_false_values=True).lower()
    if scheme not in {"http", "https"}:
        scheme = "https"

    ak = os.getenv("VOLCENGINE_ACCESS_KEY")
    sk = os.getenv("VOLCENGINE_SECRET_KEY")
    header = {}

    if not (ak and sk):
        logger.debug(
            "Get AK/SK from environment variables failed. Try to use credential from Iam."
        )
        credential = get_credential_from_vefaas_iam()
        ak = credential.access_key_id
        sk = credential.secret_access_key
        header = {"X-Security-Token": credential.session_token}
    else:
        logger.debug("Successfully get AK/SK from environment variables.")

    request_body = {
        "SpaceId": space_id,
    }
    if agent_id:
        request_body["Id"] = agent_id
    if name:
        request_body["Name"] = name
    if runtime_id:
        request_body["RuntimeId"] = runtime_id

    result = ve_request(
        request_body=request_body,
        action="GetA2aAgent",
        ak=ak,
        sk=sk,
        service=service,
        version=version,
        region=region,
        host=host,
        header=header,
        scheme=scheme,  # noqa
    )

    agent = result.get("Result", {})
    return agent


def list_remote_a2a_agents(a2a_space_id: str, a2a_space_config: Optional[dict] = None):
    a2a_space_config = a2a_space_config or {}

    agent_list = list_a2a_agents(
        space_id=a2a_space_id,
        project_name=a2a_space_config.get("project_name", "default"),
        page_number=a2a_space_config.get("page_number", 1),
        page_size=a2a_space_config.get("page_size", 10),
        version=a2a_space_config.get("version", "2025-10-30"),
    )

    agentkit_a2a_agents = []
    for agent_info in agent_list:
        a2a_agent_id = agent_info.get("Id")
        agent = get_a2a_agent(space_id=a2a_space_id, agent_id=a2a_agent_id)
        if agent.get("Status", "NotFound").lower() != "running":
            logger.warning(f"A2A Agent {a2a_agent_id} is not running, skipped.")
            continue

        agent_card = json.loads(agent.get("AgentCard"))

        agentkit_a2a_agents.append(
            # RemoteVeAgent(
            #     name=agent_info.get("Name", ""),
            #     url=agent_info.get("Host", ""),
            #     # auth_method="",
            #     # auth_token=""
            # )
            agent_card
        )
    return agentkit_a2a_agents
