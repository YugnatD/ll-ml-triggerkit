"""ll-ml-triggerkit: trainable low-level ML trigger chains for Cherenkov cameras.

Import the package name is ``triggerkit`` (the distribution on PyPI is
``ll-ml-triggerkit``). The heavy pieces (TensorFlow, ctapipe) are pulled in only
when you import the submodules that need them -- importing ``triggerkit`` itself
stays light.
"""

__version__ = "0.1.0"
