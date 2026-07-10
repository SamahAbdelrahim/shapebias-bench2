"""Import all local model wrappers so they auto-register."""

from . import smolvlm  # noqa: F401
from . import internvl  # noqa: F401
# tinyllava is deprecated — incompatible with current transformers
from . import qwen  # noqa: F401
from . import qwen35  # noqa: F401
