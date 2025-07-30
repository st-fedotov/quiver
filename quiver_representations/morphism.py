from __future__ import annotations
import numpy as np
from typing import Dict, List, Tuple, Union, Optional, Any
import json
import copy

from .field import Field, ZeroMap
from .quiver import Quiver

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .module import Module

