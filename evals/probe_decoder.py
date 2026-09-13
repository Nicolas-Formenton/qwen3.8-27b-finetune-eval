from bfcl_eval.model_handler.utils import default_decode_ast_prompting
from bfcl_eval.constants.enums import ReturnFormat

CALL = '{"name": "calculate_triangle_area", "arguments": {"base": 10, "height": 5, "unit": "units"}}'
CANDS = {
    "bare_obj": CALL,
    "obj_list": "[" + CALL + "]",
    "tag": "<tool_call>\n" + CALL + "\n</tool_call>",
    "tag_inline": "<tool_call>" + CALL + "</tool_call>",
}

for name, text in CANDS.items():
    for tag in (False, True):
        try:
            r = default_decode_ast_prompting(text, ReturnFormat.PYTHON, tag)
            print(f"OK   {name:12s} tag={tag!s:5s} -> {r}")
        except Exception as e:
            print(f"ERR  {name:12s} tag={tag!s:5s} -> {type(e).__name__}: {str(e)[:90]}")
