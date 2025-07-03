"""Chemical constants and utilities for the tabasco project.

This module defines the canonical atom types and related mappings used throughout
the project. The atom_dim in configs/globals/globals.yaml should match len(ATOM_NAMES).
"""

# Core atom type definitions
ATOM_NAMES = [
    "C",
    "N",
    "O",
    "F",
    "S",
    "Cl",
    "Br",
    "I",
    "*",  # dummy/unknown atom
]

ATOM_COLOR_MAP = {
    "C": "black",
    "N": "blue",
    "O": "red",
    "F": "cyan",
    "S": "yellow",
    "Cl": "green",
    "Br": "orange",
    "I": "purple",
    "*": "pink",
}
