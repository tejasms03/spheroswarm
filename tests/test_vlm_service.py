"""The RobotService the framework talks to.

These pin the SURFACE their library code calls and the refusals that keep a
Sphero from being driven as if it had wheels. Counted from their source, not
guessed: `Functions/Library/` and `Functions/Utilities/PathControl/` between
them call `set_command`, `get_all_robot_pose`, `get_all_path`, `path_list`,
`get_state`, `set_state`, `set_path`, `get_path`, `stop_all`, `resume`,
`get_name_from_id`, `get_id_from_name` and `set_name`.
"""

import pytest

from vlm.service import HALT, MOVING, Refused, SpheroRobot


class FakeData:
    def __init__(self, poses=None):
        self.robot_poses = poses if poses is not None else {}


class FakeServer:
    def __init__(self, poses=None):
        self.Data = FakeData(poses)


def a_service(poses=None, attached=True):
    svc = SpheroRobot(id_list=(2,), names=["Caraxes"], codes={2: "CRXS"})
    svc._set_server_reference(FakeServer(poses))
    if attached:
        svc.attach()
    return svc


A_PATH = [[400, 300, 0.0, 0], [700, 550, 0.5, 0]]


# -- the surface their code calls --------------------------------------------

@pytest.mark.parametrize("name", [
    "get_all_robot_pose", "get_all_path", "path_list", "get_state",
    "set_state", "set_path", "get_path", "stop_all", "resume",
    "get_name_from_id", "get_id_from_name", "set_name", "set_command",
])
def test_every_method_their_library_calls_exists(name):
    assert callable(getattr(SpheroRobot(), name))


def test_path_list_is_a_METHOD_because_that_is_how_pp_calls_it():
    """`pp.py:721` writes `c.Robot.path_list()[id]`.

    Against their plain dict attribute that cannot work over RPC -- the proxy
    turns every name into a call and the server calls the dict. As a method it
    does what the line plainly means.
    """
    svc = a_service()
    svc.set_path(2, A_PATH)
    assert svc.path_list()[2] == A_PATH


def test_tip_data_sim_exists_because_pp_reads_it_in_ten_places():
    assert SpheroRobot().tip_data_sim is None


# -- poses read through from DataService --------------------------------------

def test_poses_come_from_the_data_service_the_bridge_publishes_into():
    svc = a_service({2: {"x": 400.0, "y": 300.0, "theta": 1.0}})
    assert svc.get_all_robot_pose()[2]["x"] == 400.0
    assert svc.get_pose(2) == [400.0, 300.0, 1.0]


def test_an_UNSEEN_ball_is_absent_rather_than_remembered():
    """The bridge publishes {} when the tracker is not locked, and this must
    pass that straight through. Their `PoseFilter` and `pp.get_pos` both hold a
    last pose, which makes a lost ball look exactly like a still one."""
    svc = a_service({})
    assert svc.get_all_robot_pose() == {}
    assert svc.get_pose(2) == []


def test_no_server_reference_is_no_poses_rather_than_a_crash():
    assert SpheroRobot().get_all_robot_pose() == {}


# -- the refusal that matters -------------------------------------------------

def test_set_command_is_REFUSED_not_quietly_translated():
    """`[left, right, gripper]` is differential drive. A Sphero is a ball.

    Reading (v, w) back out of a wheel pair and integrating w into a heading
    puts an open-loop dead-reckon under a controller that assumes closed-loop
    heading feedback, on a robot whose heading is not measured at all. It would
    run, and be wrong in a way that looks like tuning.
    """
    with pytest.raises(Refused) as e:
        a_service().set_command(2, [120, 60, None])
    assert "differential" in str(e.value)
    assert "set_course" in str(e.value)


def test_the_refusal_names_what_to_use_instead():
    with pytest.raises(Refused) as e:
        a_service().set_command(2, [90, 90, "open"])
    assert "set_path" in str(e.value)


# -- courses ------------------------------------------------------------------

def test_a_course_is_recorded_for_the_bench_not_sent_from_here():
    svc = a_service()
    svc.set_course(2, 450.0, 12.0)
    got = svc.take_course(2)
    assert got["heading_deg"] == pytest.approx(90.0)   # wrapped
    assert got["speed"] == 12.0


def test_a_course_is_taken_ONCE():
    """The bench applies it and it is gone. A course that kept being returned
    would be re-applied every tick after the controller had moved on."""
    svc = a_service()
    svc.set_course(2, 10.0, 5.0)
    assert svc.take_course(2) is not None
    assert svc.take_course(2) is None


def test_a_course_for_a_robot_not_on_this_rig_is_refused():
    with pytest.raises(Refused):
        a_service().set_course(99, 0.0, 5.0)


# -- drive requests -----------------------------------------------------------

def test_driving_needs_a_path_first():
    svc = a_service()
    assert "Generate path first" in svc.request_drive(2)


def test_a_drive_request_is_queued_for_the_bench():
    svc = a_service()
    svc.set_path(2, A_PATH)
    assert "Started controller thread" in svc.request_drive(2)
    assert svc.take_request(2)["want"] == "drive"


def test_a_request_is_taken_once():
    svc = a_service()
    svc.set_path(2, A_PATH)
    svc.request_drive(2)
    assert svc.take_request(2) is not None
    assert svc.take_request(2) is None


def test_NOTHING_DRIVES_when_no_bench_is_attached():
    """A service with nothing behind it must not say yes.

    An agent told "ok" waits for a ball that was never asked to move -- the
    same shape of lie as reporting `arrived` for a drive that merely stopped.
    """
    svc = a_service(attached=False)
    svc.set_path(2, A_PATH)
    got = svc.request_drive(2)
    assert "No robot is attached" in got
    assert svc.take_request(2) is None


def test_detaching_halts_and_stops_accepting_drives():
    svc = a_service()
    svc.set_path(2, A_PATH)
    svc.detach()
    assert not svc.attached
    assert svc.get_state(2) == HALT
    assert "No robot is attached" in svc.request_drive(2)


def test_an_unknown_id_is_refused_by_name():
    assert "doesn't exist" in a_service().request_drive(99)


# -- outcomes -----------------------------------------------------------------

def test_the_bench_reports_HOW_a_drive_ended_not_just_that_it_stopped():
    """"Stopped" is not "arrived", and the difference is the whole thing an
    agent needs to decide what to do next."""
    svc = a_service()
    svc.set_outcome(2, "stuck", "against a boundary after 2 attempts")
    got = svc.get_outcome(2)
    assert got["outcome"] == "stuck"
    assert "boundary" in got["reason"]


def test_an_outcome_halts_the_state():
    svc = a_service()
    svc.set_state(2, MOVING)
    svc.set_outcome(2, "arrived")
    assert svc.get_state(2) == HALT


def test_a_new_drive_clears_the_previous_outcome():
    """Otherwise the verdict from the last run answers a question about this
    one, which is how a fresh drive reports a stall it never had."""
    svc = a_service()
    svc.set_outcome(2, "stuck", "old news")
    svc.set_path(2, A_PATH)
    svc.request_drive(2)
    assert svc.get_outcome(2) is None


# -- fleet-wide ---------------------------------------------------------------

def test_stop_all_queues_a_stop_and_drops_any_pending_course():
    svc = a_service()
    svc.set_course(2, 90.0, 10.0)
    svc.set_state(2, MOVING)
    svc.stop_all()
    assert svc.take_request(2)["want"] == "stop"
    assert svc.take_course(2) is None
    assert svc.get_state(2) == HALT


def test_resume_does_not_pretend_to_restart_a_loop_that_never_stopped():
    assert "resumed" in a_service().resume()


# -- names --------------------------------------------------------------------

def test_names_map_both_ways():
    svc = a_service()
    assert svc.get_name_from_id(2) == "Caraxes"
    assert svc.get_id_from_name("caraxes") == 2
    assert svc.get_name_from_id(99) is None
    assert svc.get_id_from_name("nobody") is None


def test_the_ROSTER_CODE_resolves_too_because_that_is_what_people_type():
    assert a_service().get_id_from_name("CRXS") == 2
    assert a_service().get_id_from_name("crxs") == 2


def test_renaming_works_and_an_empty_name_is_refused():
    svc = a_service()
    svc.set_name(2, "Meleys")
    assert svc.get_name_from_id(2) == "Meleys"
    with pytest.raises(ValueError):
        svc.set_name(2, "   ")
