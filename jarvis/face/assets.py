"""Read-only inventory and deterministic fallback mapping for the prototype face."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from .state import FaceActivity, FaceExpression, FaceState


class FaceAssetError(RuntimeError):
    """Raised when the immutable prototype asset inventory is not usable."""


# Keep this explicit: a model or an arbitrary directory entry must never select
# an image.  Spaces in the original upstream filenames are intentional.
BMO_ASSET_FILES: tuple[tuple[str, str], ...] = (
    ("capturing", "capturing 01.png"),
    ("error", "error 01.png"),
    ("idle", "idle 01.png"),
    ("listening", "listen 01.png"),
    ("listening", "listen 02.png"),
    ("speaking", "speaking 01.png"),
    ("speaking", "speaking 02.png"),
    ("speaking", "speaking 03.png"),
    ("thinking", "thinking 01.png"),
    ("thinking", "thinking 02.png"),
    ("thinking", "thinking 03.png"),
    ("thinking", "thinking 04.png"),
    ("warmup", "warmup 01.png"),
)

_ACTIVITY_FILES: dict[FaceActivity, tuple[tuple[str, str], ...]] = {
    FaceActivity.IDLE: (("idle", "idle 01.png"),),
    FaceActivity.WAKE_DETECTED: (("capturing", "capturing 01.png"),),
    FaceActivity.LISTENING: (
        ("listening", "listen 01.png"),
        ("listening", "listen 02.png"),
    ),
    FaceActivity.PROCESSING: tuple(
        ("thinking", f"thinking {index:02d}.png") for index in range(1, 5)
    ),
    FaceActivity.SPEAKING: tuple(
        ("speaking", f"speaking {index:02d}.png") for index in range(1, 4)
    ),
    FaceActivity.INTERRUPTED: (("listening", "listen 01.png"), ("listening", "listen 02.png")),
    FaceActivity.ERROR: (("error", "error 01.png"),),
    FaceActivity.SHUTDOWN: (("idle", "idle 01.png"),),
}

_EXPRESSION_FILES: dict[FaceExpression, tuple[str, str]] = {
    FaceExpression.NEUTRAL: ("idle", "idle 01.png"),
    FaceExpression.HAPPY: ("listening", "listen 01.png"),
    FaceExpression.AMUSED: ("speaking", "speaking 01.png"),
    FaceExpression.CURIOUS: ("capturing", "capturing 01.png"),
    FaceExpression.CONFUSED: ("error", "error 01.png"),
    FaceExpression.CONCERNED: ("error", "error 01.png"),
    FaceExpression.EXCITED: ("capturing", "capturing 01.png"),
    FaceExpression.SLEEPY: ("idle", "idle 01.png"),
    FaceExpression.THINKING: ("thinking", "thinking 01.png"),
    FaceExpression.SURPRISED: ("capturing", "capturing 01.png"),
}


@dataclass(frozen=True, slots=True)
class FaceAsset:
    key: str
    path: Path
    frame_index: int = 0


@dataclass(frozen=True, slots=True)
class FaceAssetSet:
    """An explicit, read-only asset inventory rooted inside ``faces/``."""

    faces_root: Path
    assets: tuple[FaceAsset, ...]

    @classmethod
    def from_repository(cls, repository_root: str | Path | None = None) -> "FaceAssetSet":
        if repository_root is not None:
            supplied = Path(repository_root).expanduser().resolve()
            root = supplied if supplied.name.casefold() == "faces" else supplied / "faces"
        else:
            root = Path(__file__).resolve().parents[2] / "faces"
        root = root.resolve()
        if not root.is_dir():
            raise FaceAssetError(f"face asset directory does not exist: {root}")
        assets: list[FaceAsset] = []
        for index, (folder, filename) in enumerate(BMO_ASSET_FILES):
            path = (root / folder / filename).resolve()
            try:
                path.relative_to(root)
            except ValueError:
                raise FaceAssetError("face asset escaped the faces directory") from None
            if not path.is_file():
                raise FaceAssetError(f"missing face asset: {path}")
            assets.append(FaceAsset(f"{folder}/{filename}", path, index))
        return cls(root, tuple(assets))

    def _by_pair(self, pair: tuple[str, str]) -> FaceAsset:
        key = f"{pair[0]}/{pair[1]}"
        for asset in self.assets:
            if asset.key == key:
                return asset
        raise FaceAssetError(f"asset is not in the allowlist: {key}")

    def frames_for(self, state: FaceState | FaceActivity) -> tuple[FaceAsset, ...]:
        snapshot = state if isinstance(state, FaceState) else FaceState(activity=state)
        if (
            snapshot.activity in {FaceActivity.IDLE, FaceActivity.SHUTDOWN}
            and snapshot.expression is not FaceExpression.NEUTRAL
        ):
            return (self._by_pair(_EXPRESSION_FILES[snapshot.expression]),)
        pairs = _ACTIVITY_FILES[snapshot.activity]
        # Lifecycle imagery is authoritative.  Expression mapping is the
        # deterministic fallback for an unavailable lifecycle frame only.
        return tuple(self._by_pair(pair) for pair in pairs)

    def asset_for(self, state: FaceState | FaceActivity) -> FaceAsset:
        snapshot = state if isinstance(state, FaceState) else FaceState(activity=state)
        frames = self.frames_for(snapshot)
        if frames:
            return frames[0]
        return self._by_pair(_EXPRESSION_FILES[snapshot.expression])

    def expression_asset(self, expression: FaceExpression | str) -> FaceAsset:
        snapshot = FaceState(expression=expression)
        return self._by_pair(_EXPRESSION_FILES[snapshot.expression])

    def gallery(self) -> tuple[FaceAsset, ...]:
        return self.assets

    # Friendly provider-neutral aliases used by views and tests.
    resolve = asset_for
    frames = frames_for

    def __iter__(self) -> Iterable[FaceAsset]:
        return iter(self.assets)


def default_bmo_asset_set(repository_root: str | Path | None = None) -> FaceAssetSet:
    return FaceAssetSet.from_repository(repository_root)
