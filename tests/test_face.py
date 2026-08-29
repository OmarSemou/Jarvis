from jarvis.face.assets import BMO_ASSET_FILES, default_bmo_asset_set
from jarvis.face.controller import FaceController
from jarvis.face.state import FaceActivity, FaceExpression, FaceGaze, FaceState


def test_asset_inventory_is_explicit_and_complete():
    assets = default_bmo_asset_set()
    assert len(assets.assets) == len(BMO_ASSET_FILES) == 13
    assert all(asset.path.is_file() for asset in assets.assets)
    assert all(asset.path.suffix == ".png" for asset in assets.assets)


def test_asset_mapping_and_expression_fallback_are_deterministic():
    assets = default_bmo_asset_set()
    assert assets.asset_for(FaceActivity.IDLE).path.name == "idle 01.png"
    assert [a.path.name for a in assets.frames_for(FaceActivity.SPEAKING)] == [
        "speaking 01.png",
        "speaking 02.png",
        "speaking 03.png",
    ]
    assert assets.asset_for(FaceState(expression=FaceExpression.CONCERNED)).path.name == "error 01.png"


def test_face_controller_queues_views_without_calling_them_from_observers():
    seen = []

    class View:
        def render(self, state, asset):
            seen.append((state, asset.path.name))

        def close(self):
            pass

    controller = FaceController(view=View())
    controller.set_activity(FaceActivity.LISTENING)
    assert seen == []
    assert controller.pump() == 2
    assert seen[-1][0].activity is FaceActivity.LISTENING


def test_face_controller_ignores_stale_playback_generation_events():
    controller = FaceController()
    controller.on_playback_started(10)
    assert controller.on_playback_started(9) is None
    assert controller.on_playback_stopped(9) is None
    assert controller.snapshot.activity is FaceActivity.SPEAKING
    controller.on_playback_cancelled(10)
    assert controller.snapshot.activity is FaceActivity.INTERRUPTED


def test_face_controller_observes_robot_values_without_robot_dependency():
    controller = FaceController()
    controller.observe_robot_expression("happy")
    controller.observe_robot_gaze("left")
    assert controller.snapshot.expression is FaceExpression.HAPPY
    assert controller.snapshot.gaze is FaceGaze.LEFT


def test_importing_face_does_not_create_window_or_threads():
    # This test intentionally exercises only the package-level import surface.
    import jarvis.face as face

    assert hasattr(face, "FaceController")
