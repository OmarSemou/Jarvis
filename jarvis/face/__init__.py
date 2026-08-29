"""Provider-neutral animated-face contracts for the Jarvis prototype.

The face observes application, voice, and simulated-robot state.  It owns no
tool, motion, safety, audio, camera, or hardware authority.
"""

from .assets import FaceAsset, FaceAssetError, FaceAssetSet, default_bmo_asset_set
from .controller import FaceController
from .events import FaceStateEvent, PlaybackEvent
from .state import FaceActivity, FaceExpression, FaceGaze, FaceState

__all__ = [
    "FaceActivity",
    "FaceAsset",
    "FaceAssetSet",
    "FaceAssetError",
    "FaceController",
    "FaceExpression",
    "FaceGaze",
    "FaceState",
    "FaceStateEvent",
    "PlaybackEvent",
    "default_bmo_asset_set",
]
