# Copyright (C) 2026  Murdock Grewar
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

# argparse_tools.py

"""Tools to be used with argparse.ArgumentParser and the like."""

import argparse
import enum
import typing


class EnumAction(argparse.Action):
    """Argparse action for handling Enums."""

    def __init__(self, **kwargs) -> None:  # noqa: ANN003
        # Pop off the type value
        enum_type = kwargs.pop("type", None)

        # Ensure an Enum subclass is provided
        if enum_type is None:
            msg = "type must be assigned an Enum when using EnumAction"
            raise ValueError(msg)
        if not issubclass(enum_type, enum.Enum):
            msg = "type must be an Enum when using EnumAction"
            raise TypeError(msg)

        # Generate choices from the Enum
        kwargs.setdefault(
            "choices",
            # tuple(e.value for e in enum_type)
            tuple(e.name for e in enum_type),
        )

        super().__init__(**kwargs)

        self._enum = enum_type

    def __call__(  # noqa: D102
        self,
        parser: argparse.ArgumentParser,  # noqa: ARG002
        namespace: argparse.Namespace,
        values,  # noqa: ANN001
        option_string=None,  # noqa: ANN001, ARG002
    ) -> None:
        # Convert value back into an Enum
        # value = self._enum(values)
        value = self._enum[values]
        setattr(namespace, self.dest, value)


class MaybeEnumAction(argparse.Action):
    """Argparse action for handling Enums. Allows for the additional option 'None', useful for allowing nonspecified/automatic behaviour."""

    def __init__(self, **kwargs) -> None:  # noqa: ANN003
        # Pop off the type value
        enum_type = kwargs.pop("type", None)

        # Ensure an Enum subclass is provided
        if enum_type is None:
            msg = "type must be assigned an Enum when using EnumAction"
            raise ValueError(msg)
        if not issubclass(enum_type, enum.Enum):
            msg = "type must be an Enum when using EnumAction"
            raise TypeError(msg)

        # Generate choices from the Enum
        kwargs.setdefault(
            "choices",
            # tuple(e.value for e in enum_type)
            tuple(e.name for e in enum_type),
        )
        super().__init__(**kwargs)

        self._enum = enum_type

    def __call__(  # noqa: D102
        self,
        parser: argparse.ArgumentParser,  # noqa: ARG002
        namespace: argparse.Namespace,
        values,  # noqa: ANN001
        option_string=None,  # noqa: ANN001, ARG002
    ) -> None:
        # Convert value back into an Enum
        # value = self._enum(values)
        if values is None:
            setattr(namespace, self.dest, None)
        else:
            value = self._enum[values]
            setattr(namespace, self.dest, value)


def ContiguousSliceArgType(string: str) -> slice[int | None, int | None, None]:  # noqa: N802
    """A function that decodes an integer slice like slice(1,3) or slice(None, 4) from a string."""  # noqa: D401
    string = string.replace("(", "").replace(")", "")
    if len(string) == 0:
        raise ValueError()
    if string[-1] == ",":
        string = string[:-1]
    strings = string.split(",")
    if len(strings) != 2:
        raise ValueError()

    def _str_to_int_or_None(s: str) -> int | None:  # noqa: N802
        if s.strip().lower() == "none":
            return None
        return int(s)

    return slice(*map(_str_to_int_or_None, strings))


def FloatTupleArgTypeFactory(  # noqa: N802
    *, numfloats: int
) -> typing.Callable[[str], tuple[float, ...]]:
    """A factory for functions that take in a string representing a float tuple, such as (1.0, 1.4)."""  # noqa: D401
    if type(numfloats) is not int or numfloats < 0:
        msg = "'numfloats' should be a nonnegative integer"
        raise ValueError(msg)

    def _func(string: str) -> tuple[float, ...]:
        nonlocal numfloats
        string = string.replace("(", "").replace(")", "")
        if len(string) == 0:
            raise ValueError()
        if string[-1] == ",":
            string = string[:-1]
        strings = string.split(",")
        if len(strings) != numfloats:
            msg = "Wrong number of floats."
            raise ValueError(msg)

        def _str_to_float(s: str) -> float:
            return float(s)

        return tuple(map(_str_to_float, strings))

    return _func


def IntTupleArgTypeFactory(  # noqa: N802
    *, numints: int
) -> typing.Callable[[str], tuple[int, ...]]:
    """A factory for functions that take in a string representing an int tuple, such as (1,2,3)."""  # noqa: D401
    if type(numints) is not int or numints < 0:
        msg = "'numints' should be a nonnegative integer"
        raise ValueError(msg)

    def _func(string: str) -> tuple[int, ...]:
        nonlocal numints
        string = string.replace("(", "").replace(")", "")
        if len(string) == 0:
            raise ValueError()
        if string[-1] == ",":
            string = string[:-1]
        strings = string.split(",")
        if len(strings) != numints:
            msg = "Wrong number of ints."
            raise ValueError(msg)

        def _str_to_int(s: str) -> int:
            return int(s)

        return tuple(map(_str_to_int, strings))

    return _func


def BoolArgType(string: str) -> bool:  # noqa: N802
    """A basic bool type. Fixes errors used when trying to use bool directly as an arg type. Can be combined with MaybeType."""  # noqa: D401
    match string.lower().strip():
        case "false" | "0":
            return False
        case "true" | "1":
            return True
        case _:
            msg = f"Cannot interpret string as bool: '{string}'"
            raise ValueError(msg)


def MaybeType[T](  # noqa: N802
    func: typing.Callable[[str], T],
) -> typing.Callable[[str], T | None]:
    """Extend a type like 'str' or 'float' to also accept the 'none' keyword."""

    def f(string: str) -> T | None:
        if string.lower().strip() == "none":
            return None
        return func(string)

    return f


class ArgumentDictParserWithHiddenFixedConstants(argparse.ArgumentParser):
    """Class to act as a standard ArgumentParser, but with prefilled bits in the namespace returned by parse_args()."""

    const_dict: dict[str, typing.Any]

    def __init__(self) -> None:
        self.const_dict = {}

    def add_hidden_const(self, key: str, val: typing.Any) -> None:  # noqa: ANN401, D102
        self.const_dict[key] = val

    def parse_args_and_consts(self) -> argparse.Namespace:
        """Perform parse_args(), then inserts the defined constants into the returned namespace."""
        ns: argparse.Namespace = self.parse_args()
        d = vars(ns)
        d.update(self.const_dict)
        return ns
