from typing import Any,TypedDict,Callable,TypeAlias
from numbers import Number

class EquationPart(TypedDict,total=False):
    var: str
    deriv: list[str]
    coef: Any
    op: str | Callable
    apply_coef: str

EquationType: TypeAlias = list[EquationPart] | tuple[EquationPart, ...] | str
MultiEquationType: TypeAlias = list[EquationType] | tuple[EquationType, ...]

class ConditionsPart(TypedDict,total=False):
    loc: dict[str,Number]
    eqn: EquationType
    n_samples: int

ConditionsType: TypeAlias = list[ConditionsPart] | tuple[ConditionsPart, ...]

ModelStructureType: TypeAlias = list[list | tuple | dict] | tuple[list | tuple | dict]