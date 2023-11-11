from typing import NamedTuple, Literal

ErrorName = Literal["Input-Error", "Saving-Error", "Code-Error", "Loading-Error"]


class Error(NamedTuple):
    name: ErrorName
    exception: Exception
