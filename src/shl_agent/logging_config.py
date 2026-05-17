import logging
import sys

def configure_logging(level: str = 'INFO'):
    lvl = getattr(logging, level.upper(), logging.INFO)
    fmt = '%(asctime)s %(levelname)s %(name)s - %(message)s'
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter(fmt))
    root = logging.getLogger()
    root.setLevel(lvl)
    # remove existing handlers to avoid duplication
    if root.handlers:
        for h in list(root.handlers):
            root.removeHandler(h)
    root.addHandler(handler)
