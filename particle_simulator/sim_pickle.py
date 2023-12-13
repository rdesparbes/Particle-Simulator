from typing import (
    Any,
    Dict,
    Tuple,
    Literal,
    List,
    TypedDict,
)

AttributeType = Literal["set", "entry", "var"]
ParticlesPickle = List[Dict[str, Any]]
ParticleSettings = Dict[str, Tuple[Any, AttributeType]]
SimSettings = Dict[str, Tuple[Any, AttributeType]]
SimPickle = TypedDict(
    "SimPickle",
    {
        "particles": ParticlesPickle,
        "particle-settings": ParticleSettings,
        "sim-settings": SimSettings,
    },
)
