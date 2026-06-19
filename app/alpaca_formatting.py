from dataclasses import dataclass


@dataclass
class PromptData:
    instruction: str
    input: str
    output: str


def format_alpaca_example(item):
    if item["input"].strip():
        return (
            f"### Instruction:\n{item['instruction'].strip()}\n\n"
            f"### Input:\n{item['input'].strip()}\n\n"
            f"### Response:\n{item['output'].strip()}"
        )
    return (
        f"### Instruction:\n{item['instruction'].strip()}\n\n"
        f"### Response:\n{item['output'].strip()}"
    )


def format_alpaca_prompt_and_response(item):
    if "input" in item and item["input"].strip():
        prompt = (
            f"### Instruction:\n{item['instruction'].strip()}\n\n"
            f"### Input:\n{item['input'].strip()}\n\n"
            f"### Response:\n"
        )
    else:
        prompt = (
            f"### Instruction:\n{item['instruction'].strip()}\n\n"
            f"### Response:\n"
        )

    response = item["output"].strip()
    return prompt, response
