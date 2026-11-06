"""Define the agents that the multi-agent methods share."""

from dataclasses import dataclass
from typing import List


@dataclass
class Agent:
    """Name one agent. `role` selects the agent's prompt."""
    name: str
    role: str


def default_agents() -> List[Agent]:
    """Return the four agents in the order in which they run."""
    return [
        Agent(name="Planner", role="planner"),
        Agent(name="Critic", role="critic"),
        Agent(name="Refiner", role="refiner"),
        Agent(name="Judger", role="judger"),
    ]


__all__ = ["Agent", "default_agents"]
