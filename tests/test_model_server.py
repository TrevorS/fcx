"""Model-server singleton logic, with spawn/health stubbed out (no real model)."""

import asyncio

import pytest

import fcx.model_server as ms
from fcx.config import Config


def _cfg(tmp_path, **kw):
    return Config(
        lock_path=str(tmp_path / "model.lock"),
        model_log=str(tmp_path / "model.log"),
        pid_path=str(tmp_path / "model.pid"),
        startup_timeout=5,
        **kw,
    )


async def _true():
    return True


def test_is_local():
    assert ms.is_local(Config(base_url="http://localhost:8080/v1")) is True
    assert ms.is_local(Config(base_url="http://127.0.0.1:9/v1")) is True
    assert ms.is_local(Config(base_url="https://api.openai.com/v1")) is False


def test_server_port_from_url():
    assert ms.server_port(Config(base_url="http://localhost:8080/v1")) == 8080
    assert ms.server_port(Config(base_url="http://localhost:1234/v1")) == 1234


async def test_ensure_noop_when_healthy(tmp_path, mocker):
    mocker.patch.object(ms, "_healthy", new=lambda cfg: _true())
    spawn = mocker.patch.object(ms, "_spawn_detached")
    await ms.ensure_model_up(_cfg(tmp_path))
    spawn.assert_not_called()


async def test_ensure_remote_raises(tmp_path):
    with pytest.raises(RuntimeError, match="not local"):
        await ms.ensure_model_up(_cfg(tmp_path, base_url="https://api.openai.com/v1"))


async def test_single_spawn_under_race(tmp_path, mocker):
    """Concurrent first-callers must serialize on the flock; only one spawns."""
    state = {"up": False}

    async def healthy(cfg):
        return state["up"]

    def spawn(cfg):
        state["up"] = True  # mark healthy right after the (single) spawn

    mocker.patch.object(ms, "_healthy", new=healthy)
    mocker.patch.object(ms, "_port_in_use", new=lambda port: False)
    spy = mocker.patch.object(ms, "_spawn_detached", side_effect=spawn)

    cfg = _cfg(tmp_path)
    await asyncio.gather(*(ms.ensure_model_up(cfg) for _ in range(5)))
    assert spy.call_count == 1


async def test_port_occupied_not_ours(tmp_path, mocker):
    async def healthy(cfg):
        return False

    mocker.patch.object(ms, "_healthy", new=healthy)
    mocker.patch.object(ms, "_port_in_use", new=lambda port: True)
    mocker.patch.object(ms, "_spawn_detached", side_effect=AssertionError("must not spawn"))
    with pytest.raises(RuntimeError, match="in use but not serving"):
        await ms.ensure_model_up(_cfg(tmp_path))


def test_serves_model_accepts_loaded_model():
    cfg = Config()
    body = {"object": "list", "data": [{"id": cfg.model, "object": "model"}]}
    assert ms._serves_model(body, cfg) is True


def test_serves_model_rejects_wrong_model():
    cfg = Config()
    body = {"data": [{"id": "some/other-model"}]}
    assert ms._serves_model(body, cfg) is False


def test_serves_model_rejects_html_200():
    # An unrelated server returning 200 with a non-list body (or parsed HTML) is not our model server.
    cfg = Config()
    assert ms._serves_model("<!doctype html>", cfg) is False
    assert ms._serves_model({"data": []}, cfg) is False


def test_serves_model_external_accepts_any_listed():
    # For an unmanaged remote endpoint we only require a well-formed, non-empty models list.
    cfg = Config(base_url="https://api.openai.com/v1", manage_model=False)
    body = {"data": [{"id": "gpt-5.4"}, {"id": "whatever"}]}
    assert ms._serves_model(body, cfg) is True


def test_stop_model_no_pidfile(tmp_path):
    assert ms.stop_model(_cfg(tmp_path)) is False


def test_stop_model_dead_pid(tmp_path):
    cfg = _cfg(tmp_path)
    cfg.pid_file.write_text("999999999")  # not a live process
    assert ms.stop_model(cfg) is False
    assert not cfg.pid_file.exists()  # stale pidfile is cleaned up
