from enum import StrEnum

class RenderModes(StrEnum):
    """Rendering modes supported by the environment."""

    DISPLAY_WINDOW = "human"
    """Renders the environment in a display window."""

    RGB_ARRAY = "rgb_array"
    """Returns the rendered frame as an RGB NumPy array."""

    RGB_ARRAY_LIST = "rgb_array_list"
    """Returns a list of rendered RGB frames."""

    ANSI = "ansi"
    """Returns a text representation of the environment."""

    ANSI_LIST = "ansi_list"
    """Returns a list of text representations of the environment."""

    