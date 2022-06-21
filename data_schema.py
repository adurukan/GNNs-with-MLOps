import pydantic
import typing_extensions
from typing import Union, Optional, Sequence, Type, List, Dict, Any, TypedDict, Tuple


@pydantic.dataclasses.dataclass
class new_json:
    """The new json format"""

    node_id: str
    node: List[Dict[str, str]]
    labels: typing_extensions.TypedDict(
        "labels", {"FA_1_case_1": bool, "FA_1_case_2": bool}
    )

    transaction_val: List[Union[int, float, str]]
    transaction_time: List[Union[int, float, str]]
    further_attributes: Optional[List[Union[str, int]]]
