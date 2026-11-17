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

"""Tools to be used with click.argument and the like."""

import functools
import typing
from copy import deepcopy

import click
import click_extra as ce

# ruff: disable [missing-type-function-argument,invalid-function-name]


def command_with_option_substitutions(
    command: ce.Command,
    *,
    name: str,
    fixed_option_values: dict[str, typing.Any],
    help: str | None = None,
) -> ce.Command:
    """Wrap a click_extra.Command object and substitutes argument values with constants."""
    params = []
    for param in deepcopy(
        command
    ).params:  # deepcopy is needed here to prevent old, overwritten options from appearing in --help. They don't always appear, but they do sometimes, e.g. when creating multiple copies of the command with different fixed option values.
        if isinstance(param, ce.Option) and param.name in fixed_option_values:
            continue
        params.append(param)

    # After parsing, manually insert the values into the namespace that will be returned
    @click.pass_context
    def callback(ctx, **params):  # ruff: ignore[missing-return-type-private-function, missing-type-kwargs]
        # Override values after parsing, so users cannot change them.
        # params.update(fixed_option_values)
        return ctx.invoke(
            functools.partial(command.callback, **fixed_option_values),  # ty: ignore[invalid-argument-type]
            **params,
        )

    return ce.Command(
        name,
        context_settings=command.context_settings,
        callback=callback,
        params=params,
        help=command.help if help is None else help,
        epilog=command.epilog,
        short_help=command.short_help,
        options_metavar=command.options_metavar,
        add_help_option=command.add_help_option,
        no_args_is_help=command.no_args_is_help,
        hidden=command.hidden,
        deprecated=command.deprecated,
    )


class ContiguousSliceArgType(click.ParamType[slice]):
    """A click arg type that represents a contiguous slice of indices, like slice(1,4) or slice(None, None) or slice(3, None)."""

    name: str = "(" + click.INT.name + "," + click.INT.name + ")"

    def convert(self, value, param, ctx) -> slice:  # ruff: ignore[undocumented-public-method, unused-method-argument]

        if isinstance(value, slice):
            if value.step is not None:
                raise ValueError()
            return value
        string: str = value

        string = string.replace("(", "").replace(")", "")
        if len(string) == 0:
            raise ValueError()
        if string[-1] == ",":
            string = string[:-1]
        strings = string.split(",")
        if len(strings) != 2:
            raise ValueError()

        def _str_to_int_or_None(s: str) -> int | None:
            if s.strip().lower() == "none":
                return None
            return int(s)

        return slice(*map(_str_to_int_or_None, strings))


def FloatTupleArgTypeFactory(  # ruff: ignore[missing-return-type-undocumented-public-function]
    *, numfloats: int
):
    """Create a click argument type to take in n-tuples of floats, like (1.0, 1.4)."""
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

    T = tuple[*((float,) * numfloats)]  # ty: ignore[invalid-type-form]  # ruff: ignore[non-lowercase-variable-in-function]

    class _cls(click.ParamType[T]):  # ruff: ignore[invalid-class-name]
        name: str = "(" + (click.FLOAT.name + ",") * numfloats + ")"

        def convert(self, value, param, ctx) -> T:  # ruff: ignore[unused-method-argument]
            if isinstance(value, tuple):
                return value
            string: str = value
            return _func(string)

    return _cls()


def IntTupleArgTypeFactory(  # ruff: ignore[missing-return-type-undocumented-public-function]
    *, numints: int
):
    """Create a click argument type to take in n-tuples of ints, like (1, 2, 3)."""
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

    T = tuple[*((int,) * numints)]  # ruff: ignore[non-lowercase-variable-in-function]  # ty: ignore[invalid-type-form]

    class _cls(click.ParamType[T]):  # ruff: ignore[invalid-class-name]
        name: str = "(" + (click.INT.name + ",") * numints + ")"

        def convert(self, value, param, ctx) -> T:  # ruff: ignore[unused-method-argument]
            if isinstance(value, tuple):
                return value
            string: str = value
            return _func(string)

    return _cls()


class BoolArgType(click.ParamType[bool]):
    """A click arg type that represents a boolean (True or False)."""

    name: str = "BOOL"

    def convert(self, value, param, ctx) -> bool:  # ruff: ignore[undocumented-public-method, unused-method-argument]
        if isinstance(value, bool):
            return value
        string: str = value

        match string.lower().strip():
            case "false" | "0":
                return False
            case "true" | "1":
                return True
            case _:
                msg = f"Cannot interpret string as bool: '{string}'"
                raise ValueError(msg)


def MaybeType[T](
    instantiated_paramtype: click.ParamType[T],
) -> click.ParamType[T | None]:
    """Extend a click arg type with 'None' as a sentinel value."""
    obj = instantiated_paramtype

    class _cls(click.ParamType[T | None]):  # ty: ignore[shadowed-type-variable]  # ruff: ignore[invalid-class-name]
        name: str = "MAYBE_" + obj.name

        def convert(self, value, param, ctx) -> T | None:
            if value is None:
                return None
            if type(value) is str:
                if value.lower() == "none":
                    return None
            return obj.convert(value, param, ctx)

    return _cls()


# ruff: enable [missing-type-function-argument,invalid-function-name]
