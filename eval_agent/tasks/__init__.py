from .base import Task

# Each task pulls in its own optional backend (AlfWorld needs the `alfworld`
# package, ScienceWorld the `scienceworld` jar). Import lazily so evaluating a
# single task does not require the other environments' deps to be installed.
try:
    from .webshop import WebShopTask
except Exception:  # pragma: no cover - optional dependency
    WebShopTask = None

try:
    from .sciworld import SciWorldTask
except Exception:  # pragma: no cover - optional dependency
    SciWorldTask = None

try:
    from .alfworld import AlfWorldTask
except Exception:  # pragma: no cover - optional dependency
    AlfWorldTask = None