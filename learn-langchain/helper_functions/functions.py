from typing import Any


def parser_res(res_input: str | dict[str, Any]) -> str:
    if isinstance(res_input, str):
        return res_input
    elif isinstance(res_input, dict):
        text_content = res_input.get("text")
        if text_content is not None:
            return str(text_content)
    return f"unable to parser default {res_input}"
