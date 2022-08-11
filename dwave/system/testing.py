# Copyright 2018 D-Wave Systems Inc.
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

import warnings
import collections
import dimod
import dwave.cloud.computation
import unittest.mock as mock
import dwave_networkx as dnx

from uuid import uuid4
from tabu import TabuSampler


try:
    #Simplest and efficient choice returning low energy states:
    from greedy import SteepestDescentSampler as SubstituteSampler
except ImportError:
    from dimod import SimulatedAnnealingSampler as SubstituteSampler
import concurrent.futures
import numpy as np

def _dnx_graph(topology_type, shape, nodelist, edgelist):
    """Converts DWaveSampler's structure to a Chimera, Pegasus or Zephyr NetworkX graph.
       
       This code mirrors the to_networkx_graph method in dwave.system.samplers.DWaveSampler.
       
       Returns:
            :class:`networkx.Graph`:
                Either a Chimera lattice of shape [m, n, t], a Pegasus 
                lattice of shape [m] or a Zephyr lattice of size [m,t].

        Examples:
            This example converts a selected D-Wave system solver to a graph
            and verifies it has over 2000 nodes.

            >>> from dwave.system import DWaveSampler
            ...
            >>> sampler = DWaveSampler()
            >>> g = sampler.to_networkx_graph()      # doctest: +SKIP
            >>> len(g.nodes) > 2000                  # doctest: +SKIP
            True
    """

    if topology_type == 'chimera':
        if not (1 <= len(shape) <=3):
            raise ValueError('shape is incompatible with a chimera lattice.')
        G = dnx.chimera_graph(*shape,
                              node_list=nodelist,
                              edge_list=edgelist)
        
    elif topology_type == 'pegasus':
        if len(shape) != 1:
            raise ValueError('shape is incompatible with a pegasus lattice.')
        G = dnx.pegasus_graph(shape[0],
                              node_list=nodelist,
                              edge_list=edgelist)

    elif topology_type == 'zephyr':
        if len(shape) not in (1, 2):
            raise ValueError('shape is incompatible with a zephyr lattice.')
        G = dnx.zephyr_graph(*shape,
                             node_list=nodelist,
                             edge_list=edgelist)
    else:
        # Alternative could be to create a standard network graph and
        # issue a warning. Requires new dependency on networkx.
        raise ValueError('topology_type does not match a known'
                         'QPU architecure')
    return G
    
class MockDWaveSampler(dimod.Sampler, dimod.Structured):
    """Mock sampler modeled after DWaveSampler that can be used for tests.

    Properties and topology parameters are populated qualitatively matching
    online systems, and a placeholder sampler routine based on steepest descent
    is instantiated.

    Args:
        nodelist (iterable of ints, optional):
            List of nodes to include. When not specified, all nodes
            compatible with the default topology type and shape are
            added.

        edgelist (iterable of (int,int) tuples, optional):
            List of edges to include. When not specified, all edges
            compatible with the default topology type and shape are
            added.

        properties (dictionary, optional):
            A dictionary of properties, to update default values.

        broken_nodes (iterable of ints, optional):
            List of nodes to exclude (along with associated edges), for
            emulation of unyielded qubits. This parameter is made redundant
            by the use of nodelist.
        
        broken_edges (iterable of (int,int) tuples, optional):
            List of edges to exclude, for emulation of unyielded edges.
            This parameter is made redundant by the use of edgelist.

        topology_type (string, optional, default='chimera'):
            If 'topology' is in ``properties``, this argument is ignored
            in favour of ``properties``['topology']['type']. Otherwise
            it determines the DWaveSampler topology being emulated. Supported
            options are 'chimera', 'pegasus' or 'zephyr', with 'chimera'
            being the default. Note that for Pegasus emulation the
            fabric_only=True graph is presented.
            
        topology_shape (string, optional):
            If 'topology' is in ``properties`` then this argument is
            ignored in favour of ``properties``['topology']['shape']. Otherwise
            it determines a ``topology_type`` compatible shape parameter.
            A list of three numbers [m,n,t] for Chimera, defaulted as [4,4,4]. 
            A list of one number [m] for Pegasus, defaulted as [3].
            A list of two numbers [m,t] for Zephyr, defaulted as [2,4].
            Defaults are chosen large enough to capture lattice-specific
            structure, but small enough to allow rapid testing.
    
        parameter_warnings (bool, optional, default=True):
            The MockSampler is adaptive with respect to ``num_reads``,
            ``answer_mode`` and ``max_answers`` and ``label`` 
            parameters. By default ``initial_state`` can also be mocked, if
            dwave-greedy is installed. All other parameters are ignored and a 
            warning will be raised by default.

    Examples
        The first example creates a MockSampler without reference to a
        particular online system, with a fully-yielded Chimera C16 structure.
        The second example creates a MockSampler from the default DWaveSampler:
        nodelist, edgelist and properties are derived from the sampler.
        In each case a single qubit problem is sampled.

        >>> from dwave.system.testing import MockDWaveSampler
        >>> mock_sampler = MockDWaveSampler(topology_type='chimera',
        ...                                 topology_shape=[16,16,4])
        >>> ss = mock_sampler.sample_ising({mock_sampler.nodelist[0] : -1}, {},
        ...                                num_reads = 100)
        >>> print(ss.first.energy)
        -1
        ...
        >>> from dwave.system import DWaveSampler
        >>> sampler = DWaveSampler()
        >>> mock_sampler = MockDWaveSampler.from_qpu_sampler(sampler)
        >>> ss = mock_sampler.sample_ising({mock_sampler.nodelist[0] : -1}, {},
        ...                                num_reads = 100)
        >>> print(ss.first.energy)
        -1

    """
    # Feature suggestion - add seed as an optional input, to allow reproducibility.
    
    nodelist = None
    edgelist = None
    properties = None
    parameters = None
    def __init__(self,
                 nodelist=None, edgelist=None, properties=None,
                 broken_nodes=None, broken_edges=None,
                 topology_type=None,topology_shape=None,
                 parameter_warnings=True, **config):
        self.parameter_warnings = parameter_warnings

        #Parse or default topology dependent arguments:
        if properties is not None and 'topology' in properties:
            if ('type' not in properties['topology']
                or 'shape' not in properties['topology']):
                raise ValueError("'shape' and 'type' should be keys in "
                                 "properties['topology']")
            topology_type = properties['topology']['type']
            topology_shape = properties['topology']['shape']
        else:
            if topology_type is None:
                topology_type = 'chimera'
            shape_defaults = {'chimera' : [4,4,4],
                              'pegasus' : [3],
                              'zephyr' : [2,4]}
            if topology_type in shape_defaults:
                if topology_shape is None:
                    topology_shape = shape_defaults[topology_type]
            else:
                raise ValueError("Only 'chimera', 'pegasus' and 'zephyr' "
                                 "topologies are supported")
        self.properties = {
            'chip_id' : 'MockDWaveSampler',
            'topology' : {'type': topology_type, 'shape': topology_shape}}
        
        #Create graph object, introduce defects per input arguments
        if nodelist is not None:
            self.nodelist = nodelist.copy()
        if edgelist is not None:
            self.edgelist = edgelist.copy()
        # Note that self.to_networkx_graph would point to an inherited
        # version rather than the class method here, without topology
        # information, for clarity helper function is separated.
        solver_graph = _dnx_graph(self.properties['topology']['type'],
                                  self.properties['topology']['shape'],
                                  self.nodelist, self.edgelist)
        
        if broken_nodes is None and broken_edges is None:
            self.nodelist = sorted(solver_graph.nodes)
            self.edgelist = sorted(tuple(sorted(edge))
                                   for edge in solver_graph.edges)
        else:
            if broken_nodes is None:
                broken_nodes = []
            self.nodelist = sorted(set(solver_graph.nodes).difference(broken_nodes))
            if broken_edges == None:
                broken_edges = []
            self.edgelist = sorted(tuple(sorted((u, v))) for
                                   u, v in solver_graph.edges
                                   if u not in broken_nodes
                                   and v not in broken_nodes
                                   and (u, v) not in broken_edges
                                   and (v, u) not in broken_edges)
        #Finalize yield-dependent properties:
        self.properties.update({
            'num_qubits' : len(solver_graph),
            'qubits' : self.nodelist.copy(),
            'couplers' : self.edgelist.copy(),
            'anneal_offset_ranges' : [[-0.5,0.5] if i in self.nodelist
                                      else [0,0] for i in range(len(self.nodelist))]})
        # Non-topology-dependent properties and parameters mocked from
        # Advantage_system4.1 accessed February 10th 2022
        # with simplified lists for large parameters, and modified
        # topology arguments per MockSolver initialization:
        # See also:
        # https://docs.dwavesys.com/docs/latest/c_solver_parameters.html
        
        self.parameters = {
            
            'anneal_offsets': ['parameters'],
            'anneal_schedule': ['parameters'],
            'annealing_time': ['parameters'],
            'answer_mode': ['parameters'],
            'auto_scale': ['parameters'],
            'flux_biases': ['parameters'],
            'flux_drift_compensation': ['parameters'],
            'h_gain_schedule': ['parameters'],
            'initial_state': ['parameters'],
            'max_answers': ['parameters'],
            'num_reads': ['parameters'],
            'num_spin_reversal_transforms': ['parameters'],
            'programming_thermalization': ['parameters'],
            'readout_thermalization': ['parameters'],
            'reduce_intersample_correlation': ['parameters'],
            'reinitialize_state': ['parameters'],
            'warnings': [],
            'label': []}
        
        
        self.properties.update({
            'h_range': [-4.0, 4.0],
            'j_range': [-1.0, 1.0],
            'supported_problem_types': ['ising', 'qubo'],
            'parameters': {
                'anneal_offsets':
                'Anneal offsets for each working qubit, formatted as a list, '
                'with NaN specified for unused qubits.',
                'anneal_schedule':
                "Annealing schedule formatted as a piecewise linear list of "
                "floating-point pairs of 't' and 's'.",
                'annealing_time':
                'Quantum annealing duration, in microseconds, as a positive '
                'floating point number.',
                'answer_mode':
                "Format of returned answers, as 'histogram' or 'raw' samples.",
                'auto_scale':
                'Automatic rescaling of h and J values to their available '
                'range, as a boolean flag.',
                'flux_biases':
                'Flux biases for each working qubit, in normalized offset '
                'units, formatted as a list for all qubits.',
                'flux_drift_compensation':
                'Activation of flux drift compensation, as a boolean flag.',
                'h_gain_schedule':
                "h-gain schedule formatted as a piecewise linear list of "
                "floating-point pairs of 't' and 'g'.",
                'initial_state':
                'Initial states to use for a reverse-anneal request, as a list '
                'of qubit index and state.',
                'max_answers':
                'Maximum number of answers to return.',
                'num_reads':
                'Number of states to read (answers to return), as a positive '
                'integer.',
                'num_spin_reversal_transforms':
                'Number of spin-reversal transforms (gauge transformations) to '
                'perform.',
                'programming_thermalization':
                'Time in microseconds to wait after programming the processor '
                'in order for it to cool back to base temperature, as a '
                'positive floating point number.',
                'readout_thermalization':
                'Time in microseconds to wait after each state is read from '
                'the processor in order for it to cool back to base '
                'temperature, as a positive floating point number.',
                'reduce_intersample_correlation':
                'Addition of pauses between samples, as a boolean flag.',
                'reinitialize_state':
                'Reapplication of the initial_state for every read in reverse '
                'annealing, as a boolean flag.'},
            'vfyc': False,
            'anneal_offset_step': -0.0001500217998314891,
            'anneal_offset_step_phi0': 1.4303846404537006e-05,
            'annealing_time_range': [0.02, 83000.0],
            'default_annealing_time': 20.0,
            'default_programming_thermalization': 1000.0,
            'default_readout_thermalization': 0.0,
            'extended_j_range': [-2.0, 1.0],
            'h_gain_schedule_range': [-3.0, 3.0],
            'max_anneal_schedule_points': 50,
            'max_h_gain_schedule_points': 20,
            'num_reads_range': [1, 10000],
            'per_qubit_coupling_range': [-18.0, 15.0],
            'problem_run_duration_range': [0.0, 10000000.0],
            'problem_timing_data': {'version': '1.0.0',
                                    'typical_programming_time': 14072.88,
                                    'reverse_annealing_with_reinit_prog_time_delta': 0.0,
                                    'reverse_annealing_without_reinit_prog_time_delta': 5.55,
                                    'default_programming_thermalization': 1000.0,
                                    'default_annealing_time': 20.0,
                                    'readout_time_model': 'pwl_log_log',
                                    'readout_time_model_parameters': [0.0,
                                                                      0.7699665876947938,
                                                                      1.7242758696010096,
                                                                      2.711975459489206,
                                                                      3.1639057672764026,
                                                                      3.750276915153992,
                                                                      1.539131714800995,
                                                                      1.8726623164229292,
                                                                      2.125631787097315,
                                                                      2.332672340068556,
                                                                      2.371606651233025,
                                                                      2.3716219271760215],
                                    'qpu_delay_time_per_sample': 20.54,
                                    'reverse_annealing_with_reinit_delay_time_delta': -4.5,
                                    'reverse_annealing_without_reinit_delay_time_delta': -1.5,
                                    'default_readout_thermalization': 0.0,
                                    'decorrelation_max_nominal_anneal_time': 2000.0,
                                    'decorrelation_time_range': [500.0, 10000.0]},
            'programming_thermalization_range': [0.0, 10000.0],
            'readout_thermalization_range': [0.0, 10000.0],
            'tags': [],
            'category': 'qpu',
            'quota_conversion_rate': 1})
        
        if properties is not None:
            # provided properties overwrite defaults, note that
            # particular care should be taken with respect to
            # topology-dependent arguments:
            self.properties.update(properties)
        
    @classmethod
    def from_qpu_sampler(cls, sampler):
        return cls(properties=sampler.properties,nodelist=sampler.nodelist,edgelist=sampler.edgelist)

    @dimod.bqm_structured
    def sample(self, bqm, **kwargs):
        
        # Check kwargs compatibility with parameters and substitute sampler:
        mocked_parameters={'answer_mode',
                           'max_answers',
                           'num_reads',
                           'label'}
        if SubstituteSampler is not dimod.SimulatedAnnealingSampler:
            mocked_parameters.add('initial_state')
            # steepest greedy descent is the best substitute to
            # standardize upon. A few additional considerations as
            # dwave-greedy is modified and processor scale grows:
            # (1) For large sample sets, it would be more efficient to
            # handle ``answer_mode`` and ``max_answers`` within the
            # lower-level optimized code - just as they are handled server
            # side in QPU calls.
            # (2) We could also consider using 'large_sparse_opt' to
            # exploit fixed connectivity at large scale for current QPU
            # designs, but unlikely to be a significant inefficiency.
            
            # The fallback sampler (SA) is an inferior choice, but
            # greedy may not always be installed, unlike dimod.
        for kw in kwargs:
            if kw in self.parameters:
                if self.parameter_warnings and kw not in mocked_parameters:
                    warnings.warn(f'{kw!r} parameter is valid for DWaveSampler(), '
                                  'but not mocked in MockDWaveSampler().')
            else:
                raise ValueError(f'kwarg {kw!r} invalid for MockDWaveSampler()')

        # Timing values are for demonstration only. These could be made
        # adaptive to sampler parameters and mocked topology in principle.
        info = dict(problem_id=str(uuid4()),
                    timing={'qpu_sampling_time': 82.08,
                            'qpu_anneal_time_per_sample': 20.0,
                            'qpu_readout_time_per_sample': 41.54,
                            'qpu_access_time': 8550.28,
                            'qpu_access_overhead_time': 9340.72,
                            'qpu_programming_time': 8468.2,
                            'qpu_delay_time_per_sample': 20.54,
                            'total_post_processing_time': 1124.0,
                            'post_processing_overhead_time': 1124.0})
        label = kwargs.get('label')
        if label is not None:
            info.update(problem_label=label)

        #Special handling of flux_biases, for compatibility with virtual graphs

        flux_biases = kwargs.get('flux_biases')
        if flux_biases is not None:
            self.flux_biases_flag = True
            
            
        substitute_kwargs = {'num_reads' : kwargs.get('num_reads')}
        if substitute_kwargs['num_reads'] is None:
            substitute_kwargs['num_reads'] = 1
        
        if SubstituteSampler is not dimod.SimulatedAnnealingSampler:            
            initial_state = kwargs.get('initial_state')
            if initial_state is not None:
                # Initial state format is a list of (qubit,values)
                # value=3 denotes an unused variable (should be absent
                # from bqm). 
                # Convert to format for substitute (NB: plural key)
                substitute_kwargs['initial_states'] = (
                    np.array([pair[1] for pair in initial_state
                              if pair[1]!=3],dtype=float),
                    [pair[0] for pair in initial_state if pair[1]!=3])
                
        ss = SubstituteSampler().sample(bqm, **substitute_kwargs)
        ss.info.update(info)
        
        answer_mode = kwargs.get('answer_mode')
        if answer_mode is None or answer_mode == 'histogram':
            # Default for DWaveSampler() is 'histogram'
            ss = ss.aggregate()
        
        max_answers = kwargs.get('max_answers')
        if max_answers is not None:
            # Truncate sampleset if requested. Do not reorder (per DWaveSampler())
            ss = ss.truncate(max_answers)
        
        return ss

    def to_networkx_graph(self):
        return _dnx_graph(self.properties['topology']['type'],
                          self.properties['topology']['shape'],
                          self.nodelist, self.edgelist)
    
    
class MockLeapHybridDQMSampler:
    """Mock sampler modeled after LeapHybridDQMSampler that can be used for tests."""
    def __init__(self, **config):
        self.parameters = {'time_limit': ['parameters'],
                           'label': []}

        self.properties = {'category': 'hybrid',
                           'supported_problem_types': ['dqm'],
                           'quota_conversion_rate': 20,
                           'minimum_time_limit': [[20000, 5.0],
                                                  [100000, 6.0],
                                                  [200000, 13.0],
                                                  [500000, 34.0],
                                                  [1000000, 71.0],
                                                  [2000000, 152.0],
                                                  [5000000, 250.0],
                                                  [20000000, 400.0],
                                                  [250000000, 1200.0]],
                           'maximum_time_limit_hrs': 24.0,
                           'maximum_number_of_variables': 3000,
                           'maximum_number_of_biases': 3000000000}

    def sample_dqm(self, dqm, **kwargs):
        num_samples = 12    # min num of samples from dqm solver
        samples = np.empty((num_samples, dqm.num_variables()), dtype=int)

        for vi, v in enumerate(dqm.variables):
            samples[:, vi] = np.random.choice(dqm.num_cases(v), size=num_samples)

        return dimod.SampleSet.from_samples((samples, dqm.variables),
                                             vartype='DISCRETE',
                                             energy=dqm.energies(samples))

    def min_time_limit(self, dqm):
        # not caring about the problem, just returning the min
        return self.properties['minimum_time_limit'][0][1]
    
class MockLeapHybridSolver:

    properties = {'supported_problem_types': ['bqm'],
                  'minimum_time_limit': [[1, 1.0], [1024, 1.0],
                                         [4096, 10.0], [10000, 40.0]],
                  'parameters': {'time_limit': None},
                  'category': 'hybrid',
                  'quota_conversion_rate': 1}

    supported_problem_types = ['bqm']

    def upload_bqm(self, bqm, **parameters):
        bqm_adjarray = dimod.serialization.fileview.load(bqm)
        future = concurrent.futures.Future()
        future.set_result(bqm_adjarray)
        return future

    def sample_bqm(self, sapi_problem_id, time_limit):
        #Workaround until TabuSampler supports C BQMs
        bqm = dimod.BQM(sapi_problem_id.linear,
                                    sapi_problem_id.quadratic,
                                    sapi_problem_id.offset,
                                    sapi_problem_id.vartype)
        result = TabuSampler().sample(bqm, timeout=1000*int(time_limit))
        future = dwave.cloud.computation.Future('fake_solver', None)
        future._result = {'sampleset': result, 'problem_type': 'bqm'}
        return future
