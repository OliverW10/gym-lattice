# -*- coding: utf-8 -*-

"""Gymnasium-compatible implementation of the 2D HP lattice environment."""
# Import gym modules
from io import StringIO
import sys
from math import floor
from collections import OrderedDict
from typing import Any, Dict, List, Mapping, Optional, Tuple

import gymnasium as gym
from gymnasium import (spaces, utils, logger)
import numpy as np
from numpy.typing import NDArray

# Human-readable
ACTION_TO_STR: Dict[int, str] = {
    0 : 'L', 1 : 'D',
    2 : 'U', 3 : 'R'}

POLY_TO_INT: Dict[str, int] = {
    'H' : 1, 'P' : -1
}

class Lattice2DEnv(gym.Env):
    """A 2D HP lattice environment based on Dill and Lau (1989).

    It follows an absolute Cartesian coordinate system, the location of
    the polymer is stated independently from one another. Thus, we have
    four actions (left, right, up, and down) and a chance of collision.

    The environment will first place the initial polymer at the origin. Then,
    for each step, agents place another polymer to the lattice. An episode
    ends when all polymers are placed, i.e. when the length of the action
    chain is equal to the length of the input sequence minus 1. We then
    compute the reward using the energy minimization rule while accounting
    for the collisions and traps.

    Attributes
    ----------
    seq : str
        Non-empty sequence of hydrophobic (``H``) and polar (``P``) units.
    state : OrderedDict
        Polymer chain, mapping lattice coordinates to ``H`` or ``P``.
    actions : list of int
        Valid actions taken by the agent (0=left, 1=down, 2=up, 3=right).
    collisions : int
        Number of attempted moves into occupied coordinates.
    trapped : int
        Number of times the chain became trapped.
    grid_length : int
        Length of one side of the square grid.
    midpoint : tuple of int
        Grid coordinate at which the first polymer is placed.
    grid : numpy.ndarray
        Integer grid containing the current polymer chain.

    .. [dill1989lattice] Lau, K.F., Dill, K.A.: A lattice statistical
    mechanics model of the conformational and sequence spaces of proteins.
    Macromolecules 22(10), 3986–3997 (1989)
    """
    metadata: Dict[str, List[str]] = {'render_modes': ['human', 'ansi']}

    def __init__(self, seq: str, collision_penalty: int = -2,
                 trap_penalty: float = 0.5) -> None:
        """Initialize the lattice environment.

        Parameters
        ----------
        seq : str
            Non-empty sequence containing only ``H`` and ``P``.
        collision_penalty : int
            Negative penalty for an attempted move into an occupied cell.
            Defaults to ``-2``.
        trap_penalty : float
            Fraction in the open interval ``(0, 1)`` used to calculate the
            penalty for becoming trapped. Defaults to ``0.5``; the applied
            penalty is ``floor(len(seq) * trap_penalty)``.

        Raises
        ------
        ValueError
            If the sequence contains characters other than ``H`` or ``P``,
            or if either penalty is outside its allowed range.
        AttributeError
            If ``seq`` is not a string.
        TypeError
            If a penalty cannot be compared with its required range.
        """
        try:
            if not set(seq.upper()) <= set('HP'):
                raise ValueError("%r (%s) is an invalid sequence" % (seq, type(seq)))
            self.seq = seq.upper()
        except AttributeError:
            logger.error("%r (%s) must be of type 'str'" % (seq, type(seq)))
            raise

        try:
            if collision_penalty >= 0:
                raise ValueError("%r (%s) must be negative" %
                                 (collision_penalty, type(collision_penalty)))
            if not isinstance(collision_penalty, int):
                raise ValueError("%r (%s) must be of type 'int'" %
                                 (collision_penalty, type(collision_penalty)))
            self.collision_penalty = collision_penalty
        except TypeError:
            logger.error("%r (%s) must be of type 'int'" %
                         (collision_penalty, type(collision_penalty)))
            raise

        try:
            if not 0 < trap_penalty < 1:
                raise ValueError("%r (%s) must be between 0 and 1" %
                                 (trap_penalty, type(trap_penalty)))
            self.trap_penalty = trap_penalty
        except TypeError:
            logger.error("%r (%s) must be of type 'float'" %
                         (trap_penalty, type(trap_penalty)))
            raise

        # Grid attributes
        self.grid_length = 2 * len(seq) + 1
        self.midpoint = (len(self.seq), len(self.seq))

        # Define action-observation spaces
        self.action_space = spaces.Discrete(4)
        self.observation_space = spaces.Box(low=-2, high=1,
                                            shape=(self.grid_length, self.grid_length),
                                            dtype=np.int8)

        # Initialize values
        self.reset()

    def step(self, action: int) -> Tuple[NDArray[np.int_], int, bool, bool, Dict[str, Any]]:
        """Apply an action and return the Gymnasium step result.

        The action supplied by the agent should be an integer from 0
        to 3. In this case:
            - 0 : left
            - 1 : down
            - 2 : up
            - 3 : right
        The best way to remember this is to note that they are similar to the
        'h', 'j', 'k', and 'l' keys in vim.

        Returns the observation, reward, termination and truncation flags, and
        auxiliary information as ``(observation, reward, terminated,
        truncated, info)``.

        The observations are arranged as a :code:`numpy.ndarray` matrix, more
        suitable for agents built using convolutional neural networks. The
        'H' is represented as :code:`1`s whereas the 'P's as :code:`-1`s.
        However, for the actual chain, that is, an :code:`OrderedDict` and
        not its grid-like representation, can be accessed from
        ``info['state_chain']``.

        The reward is calculated at the end of every episode, that is, when
        the length of the chain is equal to the length of the input sequence.

        Parameters
        ----------
        action : int
            Specifies the position where the next polymer will be placed
            relative to the previous one:
                - 0 : left
                - 1 : down
                - 2 : up
                - 3 : right

        Returns
        -------
        tuple
            The lattice observation, integer reward, ``terminated`` and
            ``truncated`` flags, and an info dictionary. Truncation is always
            false for this environment.

        Raises
        ------
        ValueError
            If the action is not in the discrete action space.
        IndexError
            If a step is attempted after every polymer has been placed.
        """
        if not self.action_space.contains(action):
            raise ValueError("%r (%s) invalid" % (action, type(action)))

        self.last_action = action
        is_trapped = False # Trap signal
        collision = False  # Collision signal
        # Obtain coordinate of previous polymer
        x, y = next(reversed(self.state))
        # Get all adjacent coords and next move based on action
        adj_coords = self._get_adjacent_coords((x, y))
        next_move = adj_coords[action]
        # Detects for collision or traps in the given coordinate
        idx = len(self.state)
        if next_move in self.state:
            self.collisions += 1
            collision = True
        else:
            self.actions.append(action)
            try:
                self.state.update({next_move : self.seq[idx]})
            except IndexError:
                logger.error('All molecules have been placed! Nothing can be added to the protein chain.')
                raise

            if set(self._get_adjacent_coords(next_move).values()).issubset(self.state.keys()):
                logger.warn('Your agent was trapped! Ending the episode.')
                self.trapped += 1
                is_trapped = True

        # Set-up return values
        grid = self._draw_grid(self.state)
        self.done = True if (len(self.state) == len(self.seq) or is_trapped) else False
        reward = self._compute_reward(is_trapped, collision)
        info = {
            'chain_length' : len(self.state),
            'seq_length'   : len(self.seq),
            'collisions'   : self.collisions,
            'actions'      : [ACTION_TO_STR[i] for i in self.actions],
            'is_trapped'   : is_trapped,
            'state_chain'  : OrderedDict(self.state)
        }

        terminated = self.done
        truncated = False
        return (grid.copy(), reward, terminated, truncated, info)

    def reset(self, *, seed: Optional[int] = None,
              options: Optional[Dict[str, Any]] = None) -> Tuple[NDArray[np.int_], Dict[str, Any]]:
        """Reset the environment and return its initial observation and info.

        Parameters
        ----------
        seed : int or None, optional
            Random seed passed to the Gymnasium base environment.
        options : dict or None, optional
            Reserved for additional reset options; currently unused.

        Returns
        -------
        tuple
            The initial lattice observation and an info dictionary containing
            the initial polymer chain.
        """
        super().reset(seed=seed)
        self.state = OrderedDict({(0, 0) : self.seq[0]})
        self.actions: List[int] = []
        self.collisions = 0
        self.trapped = 0
        self.done = len(self.seq) == 1

        self.grid = np.zeros(shape=(self.grid_length, self.grid_length), dtype=int)
        # Automatically assign first element into grid
        self.grid[self.midpoint] = POLY_TO_INT[self.seq[0]]

        self.last_action = None
        info = {'state_chain': OrderedDict(self.state)}
        return self.grid.copy(), info

    def render(self, mode: str = 'human') -> Any:
        """Render the lattice to standard output or an ANSI text stream.

        Parameters
        ----------
        mode : str, optional
            ``'human'`` writes to standard output and returns ``None``;
            ``'ansi'`` writes to and returns a string buffer.

        Returns
        -------
        IO[str] or None
            The ANSI text buffer for ``'ansi'`` mode, otherwise ``None`` for
            human rendering.
        """

        outfile = StringIO() if mode == 'ansi' else sys.stdout
        # Flip so highest y-value row is printed first
        grid_display = np.flipud(self.grid).astype(str)
        desc = grid_display.tolist()

        # Convert everything to human-readable symbols
        for row_index, row in enumerate(desc):
            desc[row_index] = [
                '*' if cell == '0' else 'H' if cell == '1' else 'P' if cell == '-1' else cell
                for cell in row
            ]

        # Obtain all x-y indices of elements
        x_free, y_free = np.where(grid_display == '0')
        x_h, y_h = np.where(grid_display == '1')
        x_p, y_p = np.where(grid_display == '-1')

        # All unfilled spaces are gray
        for row, col in zip(x_free, y_free):
            desc[int(row)][int(col)] = utils.colorize(desc[int(row)][int(col)], "gray")

        # All hydrophobic molecules are bold-green
        for row, col in zip(x_h, y_h):
            desc[int(row)][int(col)] = utils.colorize(desc[int(row)][int(col)], "green", bold=True)

        # All polar molecules are cyan
        for row, col in zip(x_p, y_p):
            desc[int(row)][int(col)] = utils.colorize(desc[int(row)][int(col)], "cyan")

        # Provide prompt for last action
        if self.last_action is not None:
            outfile.write("  ({})\n".format(["Left", "Down", "Up", "Right"][self.last_action]))
        else:
            outfile.write("\n")

        # Draw desc
        outfile.write("\n".join(''.join(line) for line in desc)+"\n")

        if mode != 'human':
            return outfile

    def _get_adjacent_coords(
            self, coords: Tuple[int, int]) -> Dict[int, Tuple[int, int]]:
        """Return the four neighboring coordinates for a lattice position.

        Parameters
        ----------
        coords : tuple of int
            ``(x, y)`` coordinate of the current position.

        Returns
        -------
        dict
            Coordinates keyed by action number (0=left, 1=down, 2=up,
            3=right).
        """
        x, y = coords
        adjacent_coords = {
            0 : (x - 1, y),
            1 : (x, y - 1),
            2 : (x, y + 1),
            3 : (x + 1, y),
        }

        return adjacent_coords

    def _draw_grid(
            self, chain: Mapping[Tuple[int, int], str]) -> NDArray[np.int_]:
        """Draw and return the grid representation of a polymer chain.

        Parameters
        ----------
        chain : OrderedDict
            Current chain, mapping coordinates to polymer symbols.

        Returns
        -------
        numpy.ndarray
            Vertically flipped grid of shape ``(grid_length, grid_length)``.
        """
        self.grid.fill(0)
        for coord, poly in chain.items():
            trans_x, trans_y = tuple(sum(x) for x in zip(self.midpoint, coord))
            # Recall that a numpy array works by indexing the rows first
            # before the columns, that's why we interchange.
            self.grid[(trans_y, trans_x)] = POLY_TO_INT[poly]

        return np.flipud(self.grid)

    def _compute_reward(self, is_trapped: bool, collision: bool) -> int:
        """Compute the integer reward for the current time step.

        For every timestep, we compute the reward using the following function:

        .. code-block:: python

            reward_t = state_reward 
                       + collision_penalty
                       + actual_trap_penalty

        The :code:`state_reward` is only computed at the end of the episode
        (Gibbs free energy) and its value is :code:`0` for every timestep
        before that.

        The :code:`collision_penalty` is given when the agent makes an invalid
        move, i.e. going to a space that is already occupied.

        The :code:`actual_trap_penalty` is computed whenever the agent
        completely traps itself and has no more moves available. Overall, we
        still compute for the :code:`state_reward` of the current chain but
        subtract that with the following equation:
        ``floor(len(seq) * trap_penalty)``.

        Parameters
        ----------
        is_trapped : bool
            Whether the last action left the chain trapped.
        collision : bool
            Whether the last action targeted an occupied coordinate.

        Returns
        -------
        int
            State reward adjusted by collision and trap penalties.
        """
        state_reward = self._compute_free_energy(self.state) if self.done else 0
        collision_penalty = self.collision_penalty if collision else 0
        actual_trap_penalty = -floor(len(self.seq) * self.trap_penalty) if is_trapped else 0

        # Compute reward at timestep, the state_reward is originally
        # negative (Gibbs), so we invert its sign.
        reward = - state_reward + collision_penalty + actual_trap_penalty

        return reward

    def _compute_free_energy(
            self, chain: Mapping[Tuple[int, int], str]) -> int:
        """Compute the negative Gibbs energy score for a lattice state.

        This score is computed from non-consecutive adjacent hydrophobic pairs
        using the energy function described by Dill and Lau (1989).

        The returned value is the negated Gibbs energy, so larger values
        correspond to more favorable configurations.

        .. [dill1989lattice] Lau, K.F., Dill, K.A.: A lattice statistical
        mechanics model of the conformational and sequence spaces of proteins.
        Macromolecules 22(10), 3986–3997 (1989)

        Parameters
        ----------
        chain : OrderedDict
            Current chain, mapping coordinates to polymer symbols.

        Returns
        -------
        int
            Negative Gibbs energy score of the supplied chain.
        """
        h_polymers = [x for x in chain if chain[x] == 'H']
        h_pairs = [(x, y) for x in h_polymers for y in h_polymers]

        # Compute distance between all hydrophobic pairs
        h_adjacent = []
        for pair in h_pairs:
            dist = np.linalg.norm(np.subtract(pair[0], pair[1]))
            if dist == 1.0: # adjacent pairs have a unit distance
                h_adjacent.append(pair)

        # Get the number of consecutive H-pairs in the string,
        # these are not included in computing the energy
        h_consecutive = 0
        for i in range(1, len(self.state)):
            if (self.seq[i] == 'H') and (self.seq[i] == self.seq[i-1]):
                h_consecutive += 1

        # Remove duplicate pairs of pairs and subtract the
        # consecutive pairs
        nb_h_adjacent = len(h_adjacent) / 2
        gibbs_energy = nb_h_adjacent - h_consecutive
        reward = - gibbs_energy
        return int(reward)
