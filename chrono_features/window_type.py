# ruff: noqa: D102

from abc import ABC, abstractmethod


class WindowBase(ABC):
    """Base class for all window types."""

    @property
    @abstractmethod
    def suffix(self) -> str:
        """Returns the suffix used to identify the window type.

        Returns:
            str: A string representing the window type suffix.
        """
        raise NotImplementedError

    def add_suffix_to_feature(self, feature_name: str) -> str:
        """Creates a new feature name by appending the window type suffix.

        Args:
            feature_name (str): The original feature name.

        Returns:
            str: The new feature name with the window type suffix appended.
        """
        return f"{feature_name}_{self.suffix}"

    @abstractmethod
    def __str__(self) -> str:
        """Returns a string representation of the window type.

        Returns:
            str: A string describing the window type.
        """
        raise NotImplementedError

    @abstractmethod
    def __repr__(self) -> str:
        """Returns an unambiguous string representation of the window type.

        Returns:
            str: A string that could be used to recreate the object.
        """
        raise NotImplementedError


class WindowType:
    """Class containing various types of windows."""

    class EXPANDING(WindowBase):
        """An expanding window."""

        @property
        def suffix(self) -> str:
            return "expanding"

        def __str__(self) -> str:
            return "ExpandingWindow()"

        def __repr__(self) -> str:
            return "WindowType.EXPANDING()"

    class ROLLING(WindowBase):
        """A window with a fixed size."""

        def __init__(self, size: int, *, only_full_window: bool = True) -> None:
            """Initializes the ROLLING window.

            Args:
                size (int): The size of the window.
                only_full_window (bool, optional): Whether to use only full windows. Defaults to True.

            Raises:
                ValueError: If size is less than or equal to 0
            """
            if size <= 0:
                msg = "Window size must be greater than 0"
                raise ValueError(msg)
            self.size = size
            self.only_full_window = only_full_window

        @property
        def suffix(self) -> str:
            return f"rolling_{self.size}"

        def __str__(self) -> str:
            return f"RollingWindow(size={self.size}, only_full_window={self.only_full_window})"

        def __repr__(self) -> str:
            return f"WindowType.ROLLING(size={self.size}, only_full_window={self.only_full_window})"

    class DYNAMIC(WindowBase):
        """A window with a dynamic size based on a column."""

        def __init__(self, len_column_name: str) -> None:
            self.len_column_name = len_column_name

        @property
        def suffix(self) -> str:
            return f"dynamic_based_on_{self.len_column_name}"

        def __str__(self) -> str:
            return f"DynamicWindow(len_column_name='{self.len_column_name}')"

        def __repr__(self) -> str:
            return f"WindowType.DYNAMIC(len_column_name='{self.len_column_name}')"
