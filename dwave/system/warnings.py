# Copyright 2019 D-Wave Systems Inc.
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.
#
# =============================================================================
import enum
import logging

import six
import dimod

from dwave.embedding import broken_chains


class WarningAction(enum.Enum):
    IGNORE = 'ignore'
    SAVE = 'save'

    # we may eventually want to support logging and raising python Warnings
    # LOG = 'log'
    # RAISE = 'raise'


IGNORE = WarningAction.IGNORE
SAVE = WarningAction.SAVE
# LOG = WarningAction.LOG
# RAISE = WarningAction.RAISE


class ChainBreakWarning(UserWarning):
    pass


class ChainLengthWarning(UserWarning):
    pass


class ChainStrengthWarning(UserWarning):
    """Base category for warnings about the embedding chain strength."""
    pass


class EnergyScaleWarning(UserWarning):
    """Base category for warnings about the relative bias strengths."""
    pass


class WarningHandler(object):
    def __init__(self, action=None):
        self.saved = []

        if action is None:
            action = self.default_action
        elif isinstance(action, WarningAction):
            pass
        elif isinstance(action, six.string_types):
            action = WarningAction[action.upper()]
        else:
            raise TypeError('unknown warning action provided')

        self.action = action

    default_action = WarningAction.IGNORE

    # todo: let user override __init__ parameters with kwargs
    def issue(self, msg, category=None, func=None, level=logging.WARNING,
              data=None):
        """Issue a warning.

        Args:
            msg (str):
                The warning message

            category (Warning):
                The warning category class. Defaults to UserWarning.

            level (int):
                The level of warning severity. Uses the logging warning levels.

            func (function):
                A function that is executed in the case that the warning level
                is not IGNORE. The function should return a 2-tuple containing
                a bool specifying whether the warning should be saved/raised
                and any relevent data associated with the warning as a
                dictionary/None. This overrides anything provided in the `data`
                kwarg.

            data (dict):
                Any data relevent to the warning.

        """

        if self.action is IGNORE:
            return

        if func is not None:
            valid, data = func()
            if not valid:
                return

        if category is None:
            category = UserWarning

        if data is None:
            data = {}

        if self.action is SAVE:
            self.saved.append(dict(type=category,
                                   message=msg,
                                   level=level,
                                   data=data))
        else:
            raise TypeError("unknown action")

    # some hard-coded warnings for convenience or for expensive operations

    def chain_length(self, embedding, length=7):
        if self.action is IGNORE:
            return

        for v, chain in embedding.items():
            if len(chain) <= length:
                continue

            self.issue("Chain length greater than {}".format(length),
                       category=ChainLengthWarning,
                       data=dict(target_variables=chain,
                                 source_variables=[v]),
                       )

    def chain_break(self, sampleset, embedding):
        if self.action is IGNORE:
            return

        ground = sampleset.lowest()
        variables = list(embedding)
        chains = [embedding[v] for v in variables]
        broken = broken_chains(ground, chains)

        if not (len(sampleset) and broken.any()):
            return

        for nc, chain in enumerate(chains):
            for row in range(broken.shape[0]):
                if not broken[row, nc]:
                    continue

                self.issue("A chain is broken in the lowest energy samples found",
                           category=ChainBreakWarning,
                           level=logging.ERROR,
                           data=dict(target_variables=chain,
                                     source_variables=[variables[nc]],
                                     sample_index=row),
                           )

    def chain_strength(self, bqm, chain_strength):
        """Issues a warning when any quadratic biases are greater than the given
        chain strength."""
        if self.action is IGNORE:
            return

        interactions = [uv for uv, bias in bqm.quadratic.items()
                        if abs(bias) >= chain_strength]

        if interactions:
            self.issue("Some quadratic biases are stronger than the given "
                       "chain strength",
                       category=ChainStrengthWarning,
                       level=logging.WARNING,
                       data=dict(source_interactions=interactions))

    def energy_scale(self, bqm):
        """Issues a warning if some biases are 10^3 times stronger than others.

        Args:
            bqm (:class:`dimod.BinaryQuadraticModel`/tuple):
                A binary quadratic model, a tuple of the form `(Q)` where `Q`
                is a QUBO-dictionary, or a tuple of the form `(h, J)` where
                `h` and `J` are Ising problem dictionaries.

        """
        if self.action is IGNORE:
            return

        if isinstance(bqm, tuple):
            if len(bqm) == 1:
                bqm = dimod.BinaryQuadraticModel.from_qubo(*bqm)
            elif len(bqm) == 2:
                bqm = dimod.BinaryQuadraticModel.from_ising(*bqm)
            else:
                raise TypeError("bqm should be a binary quadratic model, a "
                                "1-tuple or a 2-tuple")

        max_bias = max(map(abs, bqm.linear.values()))
        if bqm.quadratic:
            max_bias = max(max_bias, max(map(abs, bqm.quadratic.values())))

        max_bias *= 10**-3

        variables = [v for v, bias in bqm.linear.items()
                     if abs(bias) < max_bias]
        interactions = [uv for uv, bias in bqm.quadratic.items()
                        if abs(bias) < max_bias]

        data = dict()
        if variables:
            data.update(source_variables=variables)
        if interactions:
            data.update(source_interactions=interactions)

        if data:
            self.issue("Some biases are 10^3 times stronger than others",
                       category=EnergyScaleWarning,
                       level=logging.WARNING,
                       data=data)
