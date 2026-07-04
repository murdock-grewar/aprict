# argparse_tools.py

import argparse
import enum
import typing

class EnumAction(argparse.Action):
    """
    Argparse action for handling Enums
    """
    def __init__(self, **kwargs):
        # Pop off the type value
        enum_type = kwargs.pop("type", None)

        # Ensure an Enum subclass is provided
        if enum_type is None:
            raise ValueError("type must be assigned an Enum when using EnumAction")
        if not issubclass(enum_type, enum.Enum):
            raise TypeError("type must be an Enum when using EnumAction")

        # Generate choices from the Enum
        kwargs.setdefault("choices",
            # tuple(e.value for e in enum_type)
            tuple(e.name for e in enum_type)
        )

        super(EnumAction, self).__init__(**kwargs)

        self._enum = enum_type

    def __call__(self, parser, namespace, values, option_string=None):
        # Convert value back into an Enum
        # value = self._enum(values)
        value = self._enum[values]
        setattr(namespace, self.dest, value)

class MaybeEnumAction(argparse.Action):
    """
    Argparse action for handling Enums
    """
    def __init__(self, **kwargs):
        # Pop off the type value
        enum_type = kwargs.pop("type", None)

        # Ensure an Enum subclass is provided
        if enum_type is None:
            raise ValueError("type must be assigned an Enum when using EnumAction")
        if not issubclass(enum_type, enum.Enum):
            raise TypeError("type must be an Enum when using EnumAction")

        # Generate choices from the Enum
        kwargs.setdefault("choices",
            # tuple(e.value for e in enum_type)
            tuple(e.name for e in enum_type)
        )
        super(MaybeEnumAction, self).__init__(**kwargs)

        self._enum = enum_type

    def __call__(self, parser, namespace, values, option_string=None):
        # Convert value back into an Enum
        # value = self._enum(values)
        if values is None:
            setattr(namespace, self.dest, None)
        else:
            value = self._enum[values]
            setattr(namespace, self.dest, value)

def ContiguousSliceArgType(string) -> slice:
    string = string.replace("(", "").replace(")", "")
    if len(string) == 0:
        raise ValueError()
    if string[-1] == ',':
        string = string[:-1]
    strings = string.split(",")
    if len(strings) != 2:
        raise ValueError()
    def str_to_int_or_None(s: str):
        if s.strip().lower() == 'none':
            return None
        else:
            return int(s)
    return slice(*map(str_to_int_or_None, strings))

def FloatTupleArgTypeFactory(*, numfloats: int):
    if type(numfloats) is not int or numfloats < 0:
        raise ValueError("'numfloats' should be a nonnegative integer")
    def func(string) -> typing.Tuple[typing.Sequence[float]]:
        nonlocal numfloats
        string = string.replace("(", "").replace(")", "")
        if len(string) == 0:
            raise ValueError()
        if string[-1] == ',':
            string = string[:-1]
        strings = string.split(",")
        if len(strings) != numfloats:
            raise ValueError('Wrong number of floats.')
        def str_to_float(s: str):
            return float(s)
        return tuple(map(str_to_float, strings))
    return func

def IntTupleArgTypeFactory(*, numints: int):
    if type(numints) is not int or numints < 0:
        raise ValueError("'numints' should be a nonnegative integer")
    def func(string) -> typing.Tuple[typing.Sequence[int]]:
        nonlocal numints
        string = string.replace("(", "").replace(")", "")
        if len(string) == 0:
            raise ValueError()
        if string[-1] == ',':
            string = string[:-1]
        strings = string.split(",")
        if len(strings) != numints:
            raise ValueError('Wrong number of ints.')
        def str_to_int(s: str):
            return int(s)
        return tuple(map(str_to_int, strings))
    return func

def BoolArgType(string) -> bool:
    match string.lower().strip():
        case "false" | "0":
            return False
        case "true" | "1":
            return True
        case _:
            raise ValueError(f"Cannot interpret string as bool: '{string}'")

def MaybeType[T](func: typing.Callable[[str], T]) -> typing.Callable[[str], typing.Optional[T]]:
    "Extend a type like 'str' or 'float' to also accept the 'none' keyword."
    def f(string: str) -> typing.Optional[T]:
        if string.lower().strip() == 'none':
            return None
        return func(string)
    return f

class ArgumentDictParserWithHiddenFixedConstants(argparse.ArgumentParser):
    const_dict: typing.Dict[str, typing.Any] = dict()
    def add_hidden_const(self, key: str, val: typing.Any) -> None:
        self.const_dict[key] = val
    def parse_args_and_consts(self) -> argparse.Namespace:
        ns : argparse.Namespace = self.parse_args()
        d = vars(ns)
        d.update(self.const_dict)
        return ns

