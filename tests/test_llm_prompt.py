import numpy as np
import pytest

from llm.prompt import (RULES, build_system_prompt, estimate_tokens,
                        formation_lines, robot_lines)


@pytest.fixture
def ctx(sim_ctx):
    return sim_ctx(n=6, seed=3)


# -- budget -----------------------------------------------------------------
#
# Raised from 800 when the motion toolkit landed: the prompt has to carry the
# dragons' names, continuous motion, the convenience tools and docking, which
# together cost ~440 tokens and took it from 575 to ~1020. Deliberate, not
# drift. It is still small next to the ~2,660 tokens of tool schemas sent
# alongside it, and the ceiling exists to catch growth nobody chose.
BUDGET = 1200

def test_prompt_stays_under_the_token_budget(ctx):
    p = build_system_prompt(ctx)
    assert estimate_tokens(p) < BUDGET, f"prompt is {estimate_tokens(p)} tokens"


def test_prompt_budget_holds_with_a_full_library(ctx):
    for i in range(12):
        ctx.library.save_formation(f"shape{i}", ctx.active_positions(), ctx.active_codes())
    assert estimate_tokens(build_system_prompt(ctx)) < BUDGET


# -- the compute_points push -------------------------------------------------

def test_prompt_pushes_compute_points_as_primary(ctx):
    p = build_system_prompt(ctx)
    assert "compute_points" in p
    assert "primary" in p.lower()
    # two worked compute_points examples and one move_to example, per spec
    assert p.count("expression: points =") >= 2
    assert "assign:" in p


def test_prompt_states_the_core_rules(ctx):
    p = build_system_prompt(ctx)
    low = p.lower()
    assert "centimetre" in low
    assert "top-left" in low
    assert "code" in low
    assert "20cm apart" in low
    assert "wait_until_settled" in p
    assert "transform" in p
    assert "save_formation" in p


# -- state injection ---------------------------------------------------------

def test_state_injection_reflects_the_live_fleet(ctx):
    ctx.fleet["SSMK"].pos = np.array([37.0, 41.0])
    p = build_system_prompt(ctx)
    assert "SSMK/Seasmoke (37,41)" in p


def test_state_injection_updates_between_calls(ctx):
    ctx.fleet["SSMK"].pos = np.array([10.0, 10.0])
    first = build_system_prompt(ctx)
    ctx.fleet["SSMK"].pos = np.array([150.0, 120.0])
    second = build_system_prompt(ctx)
    assert "SSMK/Seasmoke (10,10)" in first
    assert "SSMK/Seasmoke (150,120)" in second


def test_arena_and_obstacles_are_injected(ctx):
    p = build_system_prompt(ctx)
    assert "Arena:" in p
    assert "Obstacles (keep clear)" in p
    assert "circle r15" in p


def test_no_obstacles_omits_the_obstacle_line(sim_ctx):
    ctx = sim_ctx(n=4, obstacles=[])
    assert "Obstacles" not in build_system_prompt(ctx)


def test_robot_count_is_stated_for_n(ctx):
    assert "n=6" in build_system_prompt(ctx)


def test_disconnected_robots_are_excluded_from_the_active_list(mixed_ctx):
    """A real robot the camera has lost must not be planned around.

    Listing it invites the model to allocate it a slot and then report success
    for a robot that never moved.
    """
    ctx = mixed_ctx(n_sim=5, tracked=False)
    lines = " ".join(robot_lines(ctx))
    assert "n=5" in lines
    assert "Offline, ignore: SNFR" in lines
    active_part = lines.split("Offline")[0]
    assert "SNFR (" not in active_part
    assert "SNFR" not in build_system_prompt(ctx).split("Offline")[0].split("CURRENT STATE")[1]


def test_all_disconnected_says_so_plainly(mixed_ctx):
    ctx = mixed_ctx(n_sim=0, tracked=False)
    p = build_system_prompt(ctx)
    assert "No robots are connected" in p


def test_saved_formations_are_listed(ctx):
    assert "none yet" in " ".join(formation_lines(ctx))
    ctx.library.save_formation("wedge", ctx.active_positions(), ctx.active_codes())
    assert "wedge" in " ".join(formation_lines(ctx))


# -- diffability -------------------------------------------------------------

def test_rules_can_be_swapped_for_a_variant(ctx):
    variant = build_system_prompt(ctx, rules="TERSE RULES.")
    assert variant.startswith("TERSE RULES.")
    assert "CURRENT STATE" in variant
    assert variant != build_system_prompt(ctx)


def test_extra_is_appended(ctx):
    p = build_system_prompt(ctx, extra="Be extremely brief.")
    assert p.rstrip().endswith("Be extremely brief.")


def test_default_rules_are_a_module_constant():
    assert "compute_points" in RULES


# -- obstacles are user-editable, so they are the prompt's growth risk -------

def test_obstacles_reach_the_model(ctx):
    p = build_system_prompt(ctx)
    assert "circle r15 at (180,40)" in p


def test_edited_obstacles_show_up_immediately(sim_ctx):
    """The UI mutates the same Workspace the prompt reads, so an obstacle
    added mid-session must appear on the very next turn."""
    from workspace.space import make_circle

    ctx = sim_ctx(n=6, obstacles=[])
    assert "Obstacles" not in build_system_prompt(ctx)

    ctx.ws.obstacles.append(make_circle((60, 60), 20))
    p = build_system_prompt(ctx)
    assert "circle r20 at (60,60)" in p


def test_many_obstacles_cannot_blow_the_token_budget(sim_ctx):
    from llm.prompt import MAX_LISTED_OBSTACLES
    from workspace.space import make_circle

    ctx = sim_ctx(n=6, obstacles=[make_circle((20 + i * 12, 30 + i * 9), 8)
                                   for i in range(40)])
    p = build_system_prompt(ctx)
    assert estimate_tokens(p) < BUDGET, f"{estimate_tokens(p)} tokens with 40 obstacles"
    assert "and 34 more" in p
    assert p.count("circle r8") == MAX_LISTED_OBSTACLES


# -- §7: what the motion toolkit added to the rules --------------------------

def test_prompt_teaches_that_dragons_are_robots(ctx):
    p = build_system_prompt(ctx)
    assert "dragons" in p.lower()
    assert "Seasmoke is SSMK" in p, "the name/code equivalence must be shown"


def test_robot_lines_carry_names_as_well_as_codes(ctx):
    """Users type names; a code-only prompt makes the model guess the mapping."""
    line = " ".join(robot_lines(ctx))
    for code, name in (("SSMK", "Seasmoke"), ("CRXS", "Caraxes")):
        assert f"{code}/{name}" in line


def test_offline_robots_keep_their_names_too(mixed_ctx):
    """The offline list is read by a human as often as by the model."""
    lines = " ".join(robot_lines(mixed_ctx(n_sim=5, tracked=False)))
    assert "Offline, ignore: SNFR/Sunfyre" in lines


def test_prompt_requires_a_duration_for_continuous_motion(ctx):
    p = build_system_prompt(ctx)
    assert "duration" in p
    assert "set_flow" in p and "set_path" in p and "follow" in p
    assert "motion_control" in p


def test_prompt_warns_against_settling_on_continuous_motion(ctx):
    assert "cannot settle" in build_system_prompt(ctx)


def test_prompt_prefers_the_convenience_tools_to_arithmetic(ctx):
    p = build_system_prompt(ctx)
    for tool in ("swap", "displace", "nudge", "gather", "spread", "mirror"):
        assert tool in p, f"{tool} is not mentioned"


def test_prompt_explains_docking_as_a_saved_formation(ctx):
    p = build_system_prompt(ctx)
    assert "dock" in p.lower()
    assert "list_formations" in p


# -- entities have to be in the prompt to be referable -----------------------

def _wanderer(ctx, role="target"):
    from workspace.entities import EntitySet
    ctx.ws.entities = EntitySet.from_data([
        {"id": "wanderer", "role": role,
         "shape": {"type": "circle", "center": [60, 90], "radius": 12},
         "motion": {"kind": "path", "waypoints": [[60, 90], [180, 90]],
                     "mode": "pingpong", "speed": 25}}])
    return ctx


def test_followable_entities_are_named_in_the_prompt(ctx):
    """Asked to follow something the prompt never mentions, a model invents it."""
    p = build_system_prompt(_wanderer(ctx))
    assert "wanderer" in p
    assert "Followable" in p


def test_moving_obstacles_are_listed_as_obstacles(ctx):
    p = build_system_prompt(_wanderer(ctx, role="obstacle"))
    assert "wanderer" in p
    assert "Moving obstacles" in p
    assert "Followable" not in p, "an obstacle-role entity is not followable"


def test_an_entity_that_is_both_appears_in_both_lists(ctx):
    p = build_system_prompt(_wanderer(ctx, role="both"))
    assert "Followable" in p and "Moving obstacles" in p


def test_no_entities_adds_no_lines(ctx):
    assert "Followable" not in build_system_prompt(ctx)


def test_entity_position_in_the_prompt_is_live(ctx):
    c = _wanderer(ctx)
    first = build_system_prompt(c)
    for _ in range(30):
        c.ws.step(0.1)
    assert build_system_prompt(c) != first, "entity position was frozen"


def test_entities_do_not_blow_the_token_budget(ctx):
    from workspace.entities import EntitySet
    ctx.ws.entities = EntitySet.from_data([
        {"id": f"thing{i}", "role": "both",
         "shape": {"type": "circle", "center": [20 + i, 20 + i], "radius": 8},
         "motion": {"kind": "path", "waypoints": [[20, 20], [180, 120]],
                     "mode": "loop", "speed": 20}} for i in range(20)])
    assert estimate_tokens(build_system_prompt(ctx)) < BUDGET


# -- the prompt must not advertise tools that are not being offered ----------
#
# Dispatch does not check what was offered, so a model that reads about
# `set_flow` in the rules will call it whether or not it has the schema —
# guessing the arguments and burning the call budget. Observed: "go round and
# round in the middle" matches no motion trigger word, and the model spent all
# six calls reaching for set_flow anyway.

def test_motion_rules_appear_only_when_motion_tools_do(ctx):
    from tools import registry

    with_motion = build_system_prompt(
        ctx, available=[t["name"] for t in registry.schemas("everyone orbit the centre")])
    assert "MOTION OVER TIME" in with_motion
    assert "set_flow" in with_motion

    without = build_system_prompt(
        ctx, available=[t["name"] for t in registry.schemas("form a circle")])
    assert "MOTION OVER TIME" not in without
    assert "set_flow" not in without


def test_convenience_rules_appear_only_when_those_tools_do(ctx):
    from tools import registry

    with_conv = build_system_prompt(
        ctx, available=[t["name"] for t in registry.schemas("swap SSMK and CRXS")])
    assert "swap" in with_conv and "gather" in with_conv

    without = build_system_prompt(
        ctx, available=[t["name"] for t in registry.schemas("form a circle")])
    assert "gather" not in without


def test_every_tool_named_in_the_prompt_is_actually_offered(ctx):
    """The invariant, checked across a spread of real commands."""
    from tools import registry

    for text in ("form a circle", "everyone orbit the centre for 20 seconds",
                 "swap SSMK and CRXS", "make the letter A",
                 "go round and round in the middle", "dock the dragons",
                 "nudge everyone left a bit"):
        offered = {t["name"] for t in registry.schemas(text)}
        prompt = build_system_prompt(ctx, available=offered)
        for tool in registry.BY_NAME:
            if f"`{tool}`" in prompt:
                assert tool in offered, (
                    f"{text!r}: prompt names `{tool}` but its schema is not sent")


def test_omitting_available_still_documents_everything(ctx):
    """Tests and eval variants that do not care still get the full rules."""
    p = build_system_prompt(ctx)
    assert "MOTION OVER TIME" in p and "swap" in p


# -- learned precedents appear on their own ----------------------------------

def test_a_relevant_precedent_is_injected_without_being_asked_for(ctx):
    from llm.memory import Memory

    ctx.memory = Memory(path=None)
    ctx.memory.record("make a circle", ["compute_points"], "radius 60cm",
                      save=False)

    p = build_system_prompt(ctx, command="make a circle please")
    assert "worked before" in p
    assert "compute_points" in p


def test_an_unrelated_command_injects_nothing(ctx):
    from llm.memory import Memory

    ctx.memory = Memory(path=None)
    ctx.memory.record("make a circle", ["compute_points"], save=False)
    assert "worked before" not in build_system_prompt(ctx, command="set the LED red")


def test_no_memory_at_all_changes_nothing(ctx):
    assert ctx.memory is None
    assert "worked before" not in build_system_prompt(ctx, command="make a circle")


def test_memory_cannot_blow_the_token_budget(ctx):
    from llm.memory import Memory

    ctx.memory = Memory(path=None)
    for i in range(200):
        ctx.memory.record(f"make a circle with quite a lot of extra words {i}",
                          ["compute_points", "transform", "wait_until_settled"],
                          "a long summary of what happened on that occasion",
                          save=False)
    assert estimate_tokens(build_system_prompt(ctx, command="make a circle")) < BUDGET
