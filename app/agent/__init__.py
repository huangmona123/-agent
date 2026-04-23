from __future__ import annotations

from app.agent.memory import new_thread_id


def get_graph():
    from app.agent.graph import get_graph as _get_graph

    return _get_graph()


def run_agent(*args, **kwargs):
    from app.agent.graph import run_agent as _run_agent

    return _run_agent(*args, **kwargs)


__all__ = ["get_graph", "new_thread_id", "run_agent"]
