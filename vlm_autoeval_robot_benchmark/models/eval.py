import typing as t

from vlm_autoeval_robot_benchmark.models.translation import (
    HISTORY_PREFIX,
    HISTORY_SUFFIX,
    PromptTemplate,
)
from vlm_autoeval_robot_benchmark.models.vlm import (
    VLM,
    VLMRequest,
    VLMResponse,
    create_modular_vlm_request,
    parse_vlm_responses,
)


async def run_episode(
    model: str,
    env_desc: str,
    task_desc: str,
    img_bytes_list: list[bytes],
    gripper_descriptors: list[str],
) -> tuple[
    list[VLMRequest], list[tuple[int, VLMResponse | None, str | None]], list[dict[str, t.Any]]
]:
    vlm = VLM()
    reqs: list[VLMRequest] = []
    input_history: list[tuple[str, list[bytes]]] = []
    for i in range(len(img_bytes_list)):
        # Create a history dict if there is a history
        if input_history:
            current_history: dict[str, str | list[tuple[str, list[bytes]]]] | None = dict(
                prefix=HISTORY_PREFIX,
                vlm_inputs=[input_history[-min(2, len(input_history))]],
                suffix=HISTORY_SUFFIX,
            )
        else:
            current_history = None

        # add request to list
        reqs.append(
            create_modular_vlm_request(
                model,
                img_bytes_list[i],
                prompt_template=PromptTemplate(
                    env_desc=env_desc,
                    task_desc=task_desc,
                    gripper_position=gripper_descriptors[i],
                    history_flag=True,
                ),
                history_dict=current_history,
            )
        )
        # accumulate history
        input_history.append(("Historical image", [img_bytes_list[i]]))

    # submit and parse responses
    responses = await vlm.generate_parallel(reqs)
    results = parse_vlm_responses(responses)
    return reqs, responses, results
