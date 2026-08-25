"""Fallback policies used by CALF-Wrapper."""

from calfwrapper.fallback.auv import AUVFallbackPolicy
from calfwrapper.fallback.base import FallbackPolicy
from calfwrapper.fallback.cartpole import CartPoleFallbackPolicy
from calfwrapper.fallback.pendulum import PendulumFallbackPolicy
from calfwrapper.fallback.robot import RobotFallbackPolicy

__all__ = [
    "AUVFallbackPolicy",
    "CartPoleFallbackPolicy",
    "FallbackPolicy",
    "PendulumFallbackPolicy",
    "RobotFallbackPolicy",
]
