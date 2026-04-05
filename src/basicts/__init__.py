from .checkpoint_compat import enable_legacy_checkpoint_compat

enable_legacy_checkpoint_compat()

from .launcher import BasicTSLauncher

__version__ = '1.1.0'

__all__ = ['__version__', 'BasicTSLauncher']
