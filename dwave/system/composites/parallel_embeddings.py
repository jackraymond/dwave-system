# Copyright 2025 D-Wave
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

"""
A :ref:`dimod composite <index_dimod>` that parallelizes small problems
on a structured sampler.

The :class:`.ParallelEmbeddingComposite` class takes a problem and
parallelizes across disjoint embeddings on a target graph.
This allows multiple independent sampling processes to be conducted in
parallel.
"""

from typing import Any

import networkx as nx

import dimod
import dwave.embedding

from dwave.embedding.utils import adjacency_to_edges, target_to_source
from minorminer.utils.parallel_embeddings import find_multiple_embeddings

__all__ = ["ParallelEmbeddingComposite"]


def _child_property_dfs(
    sampler: dimod.Sampler,
    property_name: str,
    default: Any | None = None,
    seen: set[dimod.Sampler] | None = None,
) -> Any:
    """Find a property from children

    Find some property of child solvers by depth-first search.
    Typically a QPU solver will be found at the root of the hierarchy,
    and a property such as h_range will be returned, that is otherwise
    not propagated through composition.

    This function qualitatively mirrors dimod.child_structure_dfs and may
    later be moved to that location and imported.

    Args:
        sampler:
            A :ref:`index_dimod` sampler, or a composed sampler.

        property_name:
            A property to search and return.

        default:
            Default value for a property not found.

        seen:
            IDs of already checked child samplers.

    Returns:
        The discovered property, or None if not found.

    Examples:

    >>> sampler = dimod.TrackingComposite(
    ...                 dimod.StructureComposite(
    ...                 dimod.MockDWaveSampler(), [0, 1], [(0, 1)]))
    >>> print(_child_property_dfs(sampler, property_name='j_range'))
    [-1, 1]


    """
    seen = set() if seen is None else seen

    if sampler not in seen:
        try:
            return sampler.properties[property_name]
        except (AttributeError, KeyError):
            # hasattr just tries to access anyway...
            pass

    seen.add(sampler)

    for child in getattr(sampler, "children", ()):  # getattr handles samplers
        if child in seen:
            continue

        try:
            return _child_property_dfs(child, seen=seen, property_name=property_name)
        except ValueError:
            # tree has no child samplers
            pass

    return default


class ParallelEmbeddingComposite(dimod.Composite, dimod.Structured, dimod.Sampler):
    """Parallelizes sampling of a small problem on a structured sampler.

    Enables parallel sampling on a (target) sampler, such as a quantum computer,
    by use of multiple disjoint embeddings.

    If you do not provide a list of embeddings, the function
    :func:`~minorminer.utils.parallel_embeddings.find_multiple_embeddings`,
    called by default, attempts a maximum number of embeddings. If the target
    and source graphs match :term:`QPU` architecture (e.g., :term:`Zephyr`
    :term:`topology`), the tiling of a known embedding in a regular pattern,
    such as implemented by the
    :func:`~minorminer.utils.parallel_embeddings.find_sublattice_embeddings`,
    may be a useful embedding strategy. For example, if an embedding can be
    found for a :term:`Chimera` tile, you can try an embedding of multiple
    displacements on a target QPU graph ("tiling"), and if all nodes on the
    Chimera tile are used, and the target graph is defect free, this achieves
    optimal parallelization. See the examples below and :ref:`index_minorminer`
    for information and example use cases.

    Embeddings, particularly for large subgraphs of large target graphs, can be
    difficult to find; the defaults of this composite may be slow. However, for
    QPU samplers, parallelization of job submissions may compensate for
    programming and readout times and network latency.

    Args:
       child_sampler (:class:`~dimod.Sampler`): dimod :term:`structured sampler`
            such as a :class:`~dwave.system.samplers.DWaveSampler`.

       embeddings (list, optional): A list of embeddings. Each embedding is
           assumed to be a dictionary with source-graph nodes as keys and
           iterables of target-graph nodes as values. The embeddings can include
           keys not required by the source graph. Note that ``one_to_iterable``
           is ignored (assumed True).

       source (:class:`NetworkX Graph <networkx:networkx.Graph>`, optional): A
            source graph must be provided if ``embeddings`` is not specified.
            This source graph's nodes should be supported by every embedding.

       embedder (Callable, optional): A function that returns embeddings when
            not provided. Its first two arguments are assumed to be the source
            and target graphs.

       embedder_kwargs (dict, optional): keyword arguments for the ``embedder``
            function. The default is an empty dictionary.

       one_to_iterable (bool, default=False): Defines the value type returned by
            the ``embedder`` function:

            *   False: The values in every dictionary are target nodes (defining
                a subgraph embedding). The composite transforms these to tuples
                for compatibility with the :func:`~dwave.embedding.embed_bqm`
                and :func:`~dwave.embedding.unembed_sampleset` functions.
            *   True: The values are iterables over target nodes and no
                transformation is required.

       child_structure_search (function, optional): A function that accepts a
            sampler and returns the :attr:`~dimod.Structured.structure`
            attribute. Defaults to the
            :func:`~dimod.utilities.child_structure_dfs` function.

    Raises:
        ValueError: for any of the following conditions: ``child_sampler`` is
            not structured, and the structure cannot be inferred from
            ``child_structure_search``; neither ``embeddings`` nor ``source``
            are provided; ``embeddings`` provided is an empty list, or no
            embeddings are found; ``embeddings`` and ``source`` nodes are
            inconsistent; ``embeddings`` and target graph nodes are
            inconsistent.

    Examples:

        This example submits a simple :term:`Ising` problem of just two
        variables to a quantum computer. The default subgraph embedder, the
        :func:`~minorminer.utils.parallel_embeddings.find_multiple_embeddings`
        function, is used to find a maximum number of embeddings. Note that
        searching for O(1000) of embeddings takes several seconds.

        >>> from dwave.system import DWaveSampler
        >>> from dwave.system import ParallelEmbeddingComposite
        >>> from networkx import from_edgelist
        ...
        >>> embedder_kwargs = {'max_num_emb': None}  # Without this, only 1 embedding is sought
        >>> source = from_edgelist([('a', 'b')])
        >>> with DWaveSampler() as qpu:
        ...     sampler = ParallelEmbeddingComposite(qpu, source=source, embedder_kwargs=embedder_kwargs)
        ...     sampleset = sampler.sample_ising({},{('a', 'b'): 1}, num_reads=1)
        >>> len(sampleset) > 1  # Equal to the number of parallel embeddings
        True

        Where source and target graphs have a special QPU lattice relationship
        it's possible to find an optimal parallelization through displacement.
        Note that finding a large set of disjoint chimera cells within a typical
        QPU graph can take several seconds.

        >>> from dwave.system import DWaveSampler
        >>> from dwave.system import ParallelEmbeddingComposite
        >>> from dwave_networkx import chimera_graph
        >>> from minorminer.utils.parallel_embeddings import find_sublattice_embeddings
        ...
        >>> source = tile = chimera_graph(1, 1, 4)  # A 1:1 mapping assumed
        >>> J = {e: -1 for e in tile.edges}  # A ferromagnet on the Chimera tile.
        >>> with DWaveSampler() as qpu:
        ...     embedder = find_sublattice_embeddings
        ...     embedder_kwargs = {'max_num_emb': None, 'tile': tile}
        ...     sampler = ParallelEmbeddingComposite(
        ...         qpu,
        ...         source=source,
        ...         embedder=embedder,
        ...         embedder_kwargs=embedder_kwargs)
        ...     sampleset = sampler.sample_ising({}, J, num_reads=1)
        >>> len(sampleset) > 1  # Equal to the number of parallel embeddings
        True

    See also:

        The :func:`~dwave_networkx.drawing.draw_parallel_embeddings` function to
        visualize found embeddings.

    """

    nodelist = None
    """list: Nodes of the structured sampler available to the composed sampler."""

    edgelist = None
    """list: Edges of the structured sampler available to the composed sampler."""

    parameters = None
    """dict[str, list]: Supported parameters."""

    properties = None
    """dict: Supported properties."""

    children = None
    """list [child_sampler]: List containing the structured sampler"""

    embeddings = []
    """list: Embeddings into each available tile on the structured solver."""

    def __init__(
        self,
        child_sampler,
        *,
        embeddings=None,
        source=None,
        embedder=None,
        embedder_kwargs=None,
        one_to_iterable=False,
        child_structure_search=dimod.child_structure_dfs,
        child_property_search=_child_property_dfs,
    ):
        self.parameters = child_sampler.parameters.copy()
        self.properties = properties = {"child_properties": child_sampler.properties}
        self.target_structure = child_structure_search(child_sampler)
        self.h_range = child_property_search(child_sampler, property_name="h_range")
        self.j_range = child_property_search(child_sampler, property_name="j_range")
        # dimod.Structured abstract base class automatically populates adjacency
        # and structure as mixins based on nodelist and edgelist
        if source is not None:
            self.nodelist = list(source.nodes)
            self.edgelist = list(source.edges)
        elif embeddings is None:
            raise ValueError("Either the source or embeddings must be provided")

        self.children = [child_sampler]
        target_nodelist, __, target_adjacency = self.target_structure
        if embeddings is not None:
            _embeddings = embeddings.copy()
            # Computationally cheap consistency checks, and inference of structure
            if len(_embeddings) == 0:
                raise ValueError(
                    "embeddings should be a non-empty list of dictionaries"
                )

            # Target graph consistency:
            nodelist = [v for emb in _embeddings for c in emb.values() for v in c]
            nodeset = set(nodelist)
            if len(nodelist) != len(nodeset):
                raise ValueError(
                    "embedding contains a non-disjoint embedding (target nodes reused)"
                )
            if not nodeset.issubset(target_nodelist):
                raise ValueError("embedding contains invalid target nodes")

            # Source graph consistency
            if self.nodelist is None:
                self.nodelist = list(embeddings[0].keys())
            else:
                nodeset = set(self.nodelist)
                if not all(nodeset.issubset(emb) for emb in embeddings):
                    raise ValueError(
                        "source graph is inconsistent with the embeddings specified"
                    )
            if self.edgelist is None:
                # Find the intersection graph (slow but thorough):
                edgeset = set(
                    adjacency_to_edges(
                        target_to_source(target_adjacency, embeddings[0])
                    )
                )
                for emb in embeddings[1:]:
                    edgeset0 = set(
                        adjacency_to_edges(target_to_source(target_adjacency, emb))
                    )
                    edgeset = edgeset.intersection(edgeset0)
                self.edgelist = list(edgeset)
            # could check viability of edgelist (valid embeddings), but this is slow and not the job of the composite.
        else:
            if source is None:
                raise ValueError("A source graph must be provided to infer embeddings")

            if embedder is None:
                embedder = find_multiple_embeddings
            if embedder_kwargs is None:
                embedder_kwargs = {}

            # The child_sampler may not preserve the graphical structure required
            # by the embedder. These might be passed as supplementary arguments.
            if hasattr(child_sampler, "to_networkx_graph"):
                _embeddings = embedder(
                    source, child_sampler.to_networkx_graph(), **embedder_kwargs
                )
            else:
                _embeddings = embedder(
                    source, nx.Graph(target_adjacency), **embedder_kwargs
                )

            if not one_to_iterable:
                _embeddings = [{k: (v,) for k, v in emb.items()} for emb in _embeddings]

            if len(_embeddings) == 0:
                raise ValueError(
                    "No embeddings found: consider changing the embedder or its parameters."
                )

        self.embeddings = properties["embeddings"] = _embeddings

    @dimod.bqm_structured
    def sample(
        self,
        bqm: dimod.BinaryQuadraticModel,
        chain_strength: float | None = None,
        **kwargs,
    ) -> dimod.SampleSet:
        """Sample from the specified binary quadratic model.

        Sample sets are concatenated in the same order as the ``embeddings``
        parameter used to instantiate the :class:`.ParallelEmbeddingComposite`
        class.

        Args:
            bqm:
                Binary quadratic model (:term:`BQM`) to be sampled from.

            chain_strength:
                The :term:`chain strength` parameter of the BQM.

            **kwargs:
                Optional keyword arguments for the sampling method, specified
                per embedding. Note that if :code:`auto_scale=True` (the default for
                QPU-derived composites) then all bqms are independently scaled
                to maximize the programmable range.

        Returns:
            :class:`~dimod.SampleSet`.

            The ``info`` field is returned from the child sampler unmodified.

        Examples:
            See examples in the :class:`.ParallelEmbeddingComposite` class.

        See also:
            If the ``bqm`` or ``chain_strength`` parameter varies by embedding,
            or if you want a list of sample sets as the output, use the
            :meth:`.sample_multiple` method.

        """
        chain_strengths = [chain_strength] * self.num_embeddings
        bqms = [bqm] * self.num_embeddings
        if "initial_state" in kwargs:
            kwargs["initial_states"] = [
                kwargs.pop("initial_state")
            ] * self.num_embeddings
        responses, info = self.sample_multiple(bqms, chain_strengths, **kwargs)

        if self.num_embeddings == 1:
            return responses[0]

        answer = dimod.concatenate(responses)
        answer.info.update(info)
        return answer

    def sample_multiple(
        self,
        bqms: list[dimod.BinaryQuadraticModel],
        chain_strengths: list | None = None,
        initial_states: list | None = None,
        **kwargs,
    ) -> tuple[list[dimod.SampleSet], dict]:
        """Sample from the specified binary quadratic models.

        Samplesets are returned for each binary quadratic model (:term:`BQM`) in
        the ``bqms`` list, which is submitted to the child solver with the
        corresponding item in the :term:`chain strength` and
        :ref:`initial state <parameter_qpu_initial_state>` lists, specified here
        by the ``chain_strengths`` and ``initial_states`` parameters, as well as
        the corresponding item in the :term:`minor embedding` list of the
        ``embeddings`` parameter used to instantiate the
        :class:`.ParallelEmbeddingComposite` class. These four lists should be
        ordered appropriately.

        Args:
            bqms:
                Binary quadratic models to be sampled from.

            chain_strengths:
                Chain strength per BQM.

            initial_states:
                Initial state per BQM.

            **kwargs:
                Optional keyword arguments for the sampling method.
                Note that if :code:`auto_scale=True`, each embedded BQM is
                independently scaled to maximize programmable range.

        Returns:
            Tuple: A list of :class:`~dimod.SampleSet`, one per embedding, and
            the ``info`` field returned by the child sampler.

        Examples:
            See examples in the :class:`.ParallelEmbeddingComposite` class.

        See also:
            The :meth:`.sample` method.
        """

        # apply the embeddings to the given problem to tile it across the child sampler
        embedded_bqm = dimod.BinaryQuadraticModel.empty(bqms[0].vartype)

        __, __, target_adjacency = self.target_structure
        if not chain_strengths:
            chain_strengths = [None] * self.num_embeddings

        if initial_states is not None and any(i_s for i_s in initial_states):
            kwargs["initial_state"] = {
                u: state[v]
                for embedding, state in zip(self.embeddings, initial_states)
                for v, chain in embedding.items()
                for u in chain
            }

        auto_scale = kwargs.get("auto_scale", True)
        for embedding, bqm, chain_strength in zip(
            self.embeddings, bqms, chain_strengths
        ):
            new_embedded_bqm = dwave.embedding.embed_bqm(
                bqm, embedding, target_adjacency, chain_strength=chain_strength
            )
            if auto_scale:
                # Rescale to common bounds:
                new_embedded_bqm.normalize(
                    bias_range=self.h_range, quadratic_range=self.j_range
                )
            embedded_bqm.update(new_embedded_bqm)

        # solve the problem on the child system
        tiled_response = self.child.sample(embedded_bqm, **kwargs)

        responses = []
        for embedding, bqm in zip(self.embeddings, bqms):
            responses.append(
                dwave.embedding.unembed_sampleset(tiled_response, embedding, bqm)
            )

        return responses, tiled_response.info

    @property
    def num_embeddings(self) -> int:
        """Number of embeddings available for replicating the problem."""
        return len(self.embeddings)
