
# test.py
print(__name__)

try:
    # Trying to find module in the parent package
    from . import config
    print(config.debug)
    del config
except ImportError:
    print('Relative import failed')

try:
    # Trying to find module on sys.path
    import config
    print(config.debug)
except ModuleNotFoundError:
    print('Absolute import failed')


# main.py
import ryan.test

