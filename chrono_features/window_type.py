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
        pass


class WindowType:
    """Class containing various types of windows."""

    class ROLLING(WindowBase):
        """A window with a fixed size."""

        def __init__(self, size: int, only_full_window: bool = True):
            """Initializes the ROLLING window.

            Args:
                size (int): The size of the window.
                only_full_window (bool, optional): Whether to use only full windows. Defaults to True.
            """
            self.size = size
            self.only_full_window = only_full_window

        @property
        def suffix(self) -> str:
            """Returns the suffix for the ROLLING window.

            Returns:
                str: A string representing the window type suffix.
            """
            return f"rolling_{self.size}"

    class DYNAMIC(WindowBase):
        """A window with a dynamic size based on a column."""

        def __init__(self, len_column_name: str):
            """Initializes the DYNAMIC window.

            Args:
                len_column_name (str): The name of the column that determines the window size.
            """
            self.len_column_name = len_column_name

        @property
        def suffix(self) -> str:
            """Returns the suffix for the DYNAMIC window.

            Returns:
                str: A string representing the window type suffix.
            """
            return f"dynamic_based_on_{self.len_column_name}"

    class EXPANDING(WindowBase):
        """An expanding window."""

        @property
        def suffix(self) -> str:
            """Returns the suffix for the EXPANDING window.

            Returns:
                str: A string representing the window type suffix.
            """
            return "expanding"
