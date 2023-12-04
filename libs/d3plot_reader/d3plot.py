from dataclasses import dataclass
import mmap
import os
import re
import traceback
import typing
from typing import Any, Dict, Iterable, List, Set, Tuple, Union

import numpy as np

from .binary_buffer import BinaryBuffer
from .files import open_file_or_filepath
from .array_type import ArrayType
from .d3plot_header import D3plotFiletype, D3plotHeader
from .filter_type import FilterType

# pylint: disable = too-many-lines

FORTRAN_OFFSET = 1

def _check_ndim(d3plot, array_dim_names: Dict[str, List[str]]):
    """Checks if the specified array is fine in terms of ndim

    Parameters
    ----------
    d3plot: D3plot
        d3plot holding arrays
    array_dim_names: Dict[str, List[str]]
    """

    for type_name, dim_names in array_dim_names.items():
        if type_name in d3plot.arrays:
            array = d3plot.arrays[type_name]
            if array.ndim != len(dim_names):
                msg = "Array {0} must have {1} instead of {2} dimensions: ({3})"
                dim_names_text = ", ".join(dim_names)
                raise ValueError(msg.format(type_name, len(dim_names), array.ndim, dim_names_text))


def _check_array_occurrence(
    d3plot, array_names: List[str], required_array_names: List[str]
) -> bool:
    """Check if an array exists, if all depending on it exist too

    Parameters
    ----------
    array_names: List[str]
        list of base arrays
    required_array_names: List[str]
        list of array names which would be required

    Returns
    -------
    exists: bool
        if the arrays exist or not

    Raises
    ------
    ValueError
        If a required array is not present
    """

    if any(name in d3plot.arrays for name in array_names):
        if not all(name in d3plot.arrays for name in required_array_names):
            msg = "The arrays '{0}' require setting also the arrays '{1}'"
            raise ValueError(msg.format(", ".join(array_names), ", ".join(required_array_names)))
        return True
    return False


def _negative_to_positive_state_indexes(indexes: Set[int], n_entries) -> Set[int]:
    """Convert negative indexes of an iterable to positive ones

    Parameters
    ----------
    indexes: Set[int]
        indexes to check and convert
    n_entries: int
        total number of entries

    Returns
    -------
    new_entries: Set[int]
        the positive indexes
    """

    new_entries: Set[int] = set()
    for _, index in enumerate(indexes):
        new_index = index + n_entries if index < 0 else index
        if new_index >= n_entries:
            err_msg = "State '{0}' exceeds the maximum number of states of '{1}'"
            raise ValueError(err_msg.format(index, n_entries))
        new_entries.add(new_index)
    return new_entries


@dataclass
class MemoryInfo:
    """MemoryInfo contains info about memory regions in files"""

    start: int = 0
    length: int = 0
    filepath: str = ""
    n_states: int = 0
    filesize: int = 0
    use_mmap: bool = False


class MaterialSectionInfo:
    """MaterialSectionInfo contains vars from the material section"""

    n_rigid_shells: int = 0


class SphSectionInfo:
    """SphSectionInfo contains vars from the sph geometry section"""

    n_sph_array_length: int = 11
    n_sph_vars: int = 0
    has_influence_radius: bool = False
    has_particle_pressure: bool = False
    has_stresses: bool = False
    has_plastic_strain: bool = False
    has_material_density: bool = False
    has_internal_energy: bool = False
    has_n_affecting_neighbors: bool = False
    has_strain_and_strainrate: bool = False
    has_true_strains: bool = False
    has_mass: bool = False
    n_sph_history_vars: int = 0


class AirbagInfo:
    """AirbagInfo contains vars used to describe the sph geometry section"""

    n_geometric_variables: int = 0
    n_airbag_state_variables: int = 0
    n_particle_state_variables: int = 0
    n_particles: int = 0
    n_airbags: int = 0
    # ?
    subver: int = 0
    n_chambers: int = 0

    def get_n_variables(self) -> int:
        """Get the number of airbag variables

        Returns
        -------
        n_airbag_vars: int
            number of airbag vars
        """
        return (
            self.n_geometric_variables
            + self.n_particle_state_variables
            + self.n_airbag_state_variables
        )


class NumberingInfo:
    """NumberingInfo contains vars from the part numbering section (ids)"""

    # the value(s) of ptr is initialized
    # as 1 since we need to make it
    # negative if part_ids are written
    # to file and 0 cannot do that ...
    # This is ok for self-made D3plots
    # since these fields are unused anyway
    ptr_node_ids: int = 1
    has_material_ids: bool = False
    ptr_solid_ids: int = 1
    ptr_beam_ids: int = 1
    ptr_shell_ids: int = 1
    ptr_thick_shell_ids: int = 1
    n_nodes: int = 0
    n_solids: int = 0
    n_beams: int = 0
    n_shells: int = 0
    n_thick_shells: int = 0
    ptr_material_ids: int = 1
    ptr_material_ids_defined_order: int = 1
    ptr_material_ids_crossref: int = 1
    n_parts: int = 0
    n_parts2: int = 0
    n_rigid_bodies: int = 0


@dataclass
class RigidBodyMetadata:
    """RigidBodyMetadata contains vars from the rigid body metadata section.
    This section comes before the individual rigid body data.
    """

    internal_number: int
    n_nodes: int
    node_indexes: np.ndarray
    n_active_nodes: int
    active_node_indexes: np.ndarray


class RigidBodyInfo:
    """RigidBodyMetadata contains vars for the individual rigid bodies"""

    rigid_body_metadata_list: Iterable[RigidBodyMetadata]
    n_rigid_bodies: int = 0

    def __init__(
        self, rigid_body_metadata_list: Iterable[RigidBodyMetadata], n_rigid_bodies: int = 0
    ):
        self.rigid_body_metadata_list = rigid_body_metadata_list
        self.n_rigid_bodies = n_rigid_bodies


class RigidRoadInfo:
    """RigidRoadInfo contains metadata for the description of rigid roads"""

    n_nodes: int = 0
    n_road_segments: int = 0
    n_roads: int = 0
    # ?
    motion: int = 0

    def __init__(
        self, n_nodes: int = 0, n_road_segments: int = 0, n_roads: int = 0, motion: int = 0
    ):
        self.n_nodes = n_nodes
        self.n_road_segments = n_road_segments
        self.n_roads = n_roads
        self.motion = motion


class StateInfo:
    """StateInfo holds metadata for states which is currently solely the timestep.
    We all had bigger plans in life ...
    """

    n_timesteps: int = 0

    def __init__(self, n_timesteps: int = 0):
        self.n_timesteps = n_timesteps


class D3plot:
    """Class used to read LS-Dyna d3plots"""
    _header: D3plotHeader
    _material_section_info: MaterialSectionInfo
    _sph_info: SphSectionInfo
    _airbag_info: AirbagInfo
    _numbering_info: NumberingInfo
    _rigid_body_info: RigidBodyInfo
    _rigid_road_info: RigidRoadInfo
    _buffer: Union[BinaryBuffer, None] = None

    # This amount of args is needed
    # pylint: disable = too-many-arguments, too-many-statements, unused-argument
    def __init__(
        self,
        filepath: str = None,
        n_files_to_load_at_once: Union[int, None] = None,
        state_array_filter: Union[List[str], None] = None,
        state_filter: Union[None, Set[int]] = None,
        buffered_reading: bool = False,
    ):
        """Constructor for a D3plot

        Parameters
        ----------
        filepath: str
            path to a d3plot file
        use_femzip: bool
            Not used anymore.
        n_files_to_load_at_once: int
            *DEPRECATED* not used anymore, use `buffered_reading`
        state_array_filter: Union[List[str], None]
            names of arrays which will be the only ones loaded from state data
        state_filter: Union[None, Set[int]]
            which states to load. Negative indexes count backwards.
        buffered_reading: bool
            whether to pull only a single state into memory during reading

        Examples
        --------
            >>> from lasso.dyna import D3plot, ArrayType
            >>> # open and read everything
            >>> d3plot = D3plot("path/to/d3plot")

            >>> # only read node displacement
            >>> d3plot = D3plot("path/to/d3plot", state_array_filter=["node_displacement"])
            >>> # or with nicer syntax
            >>> d3plot = D3plot("path/to/d3plot", state_array_filter=[ArrayType.node_displacement])

            >>> # only load first and last state
            >>> d3plot = D3plot("path/to/d3plot", state_filter={0, -1})

            >>> # our computer lacks RAM so lets extract a specific array
            >>> # but only keep one state at a time in memory
            >>> d3plot = D3plot("path/to/d3plot",
            >>>                 state_array_filter=[ArrayType.node_displacement],
            >>>                 buffered_reading=True)

        Notes
        -----
            If dyna wrote multiple files for several states,
            only give the path to the first file.
        """
        self._arrays = {}
        self._header = D3plotHeader()
        self._material_section_info = MaterialSectionInfo()
        self._sph_info = SphSectionInfo()
        self._airbag_info = AirbagInfo()
        self._numbering_info = NumberingInfo()
        self._rigid_body_info = RigidBodyInfo(rigid_body_metadata_list=tuple())
        self._rigid_road_info = RigidRoadInfo()
        self._state_info = StateInfo()

        # which states to load
        self.state_filter = state_filter

        # how many files to load into memory at once
        if n_files_to_load_at_once is not None:
            warn_msg = "D3plot argument '{0}' is deprecated. Please use '{1}=True'."
            raise DeprecationWarning(warn_msg.format("n_files_to_load_at_once", "buffered_reading"))
        self.buffered_reading = buffered_reading or (state_filter is not None and any(state_filter))

        # arrays to filter out
        self.state_array_filter = state_array_filter

        # load memory accordingly
        if filepath:
            self._buffer = BinaryBuffer(filepath)
            self.bb_states = None
        # no data to load basically
        else:
            self._buffer = None
            self.bb_states = None

        self.geometry_section_size = 0

        # read header
        self._read_header()

        # read geometry
        self._parse_geometry()

        # read state data
        self._read_states(filepath)

    @property
    def n_timesteps(self) -> int:
        """Number of timesteps loaded"""
        return self._state_info.n_timesteps

    @property
    def arrays(self) -> dict:
        """Dictionary holding all d3plot arrays

        Notes
        -----
            The corresponding keys of the dictionary can
            also be found in `lasso.dyna.ArrayTypes`, which
            helps with IDE integration and code safety.

        Examples
        --------
            >>> d3plot = D3plot("some/path/to/d3plot")
            >>> d3plot.arrays.keys()
            dict_keys(['irbtyp', 'node_coordinates', ...])
            >>> # The following is good coding practice
            >>> import lasso.dyna.ArrayTypes.ArrayTypes as atypes
            >>> d3plot.arrays[atypes.node_displacmeent].shape
        """
        return self._arrays

    @arrays.setter
    def arrays(self, array_dict: dict):
        assert isinstance(array_dict, dict)
        self._arrays = array_dict

    @property
    def header(self) -> D3plotHeader:
        """Instance holding all d3plot header information

        Returns
        -------
        header: D3plotHeader
            header of the d3plot

        Notes
        -----
            The header contains a lot of information such as number
            of elements, etc.

        Examples
        --------
            >>> d3plot = D3plot("some/path/to/d3plot")
            >>> # number of shells
            >>> d3plot.header.n_shells
            85624
        """
        return self._header

    @staticmethod
    def _is_end_of_file_marker(
        buffer: BinaryBuffer, position: int, ftype: Union[np.float32, np.float64]
    ) -> bool:
        """Check for the dyna eof marker at a certain position

        Parameters
        ----------
        bb: BinaryBuffer
            buffer holding memory
        position: int
            position in the buffer
        ftype: Union[np.float32, np.float64]
            floating point type

        Returns
        -------
        is_end_marker: bool
            if at the position is an end marker

        Notes
        -----
            The end of file marker is represented by a floating point
            number with the value -999999 (single precision hex: F02374C9,
            double precision hex: 000000007E842EC1).
        """

        if ftype not in (np.float32, np.float64):
            err_msg = "Floating point type '{0}' is not a floating point type."
            raise ValueError(err_msg.format(ftype))

        return buffer.read_number(position, ftype) == ftype(-999999)

    def _correct_file_offset(self):
        """Correct the position in the bytes

        Notes
        -----
            LS-Dyna writes its files zero padded at a size of
            512 words in block size. There might be a lot of
            unused trailing data in the rear we need to skip
            in order to get to the next useful data block.
        """

        if not self._buffer:
            return

        block_count = len(self._buffer) // (512 * self.header.wordsize)

        # Warning!
        # Resets the block count!
        self.geometry_section_size = (block_count + 1) * 512 * self.header.wordsize

    @property
    def _n_parts(self) -> int:
        """Get the number of parts contained in the d3plot

        Returns
        -------
        n_parts: int
            number of total parts
        """

        n_parts = (
            self.header.n_solid_materials
            + self.header.n_beam_materials
            + self.header.n_shell_materials
            + self.header.n_thick_shell_materials
            + self._numbering_info.n_rigid_bodies
        )

        return n_parts

    @property
    def _n_rigid_walls(self) -> int:
        """Get the number of rigid walls in the d3plot

        Returns
        -------
        n_rigid_walls: int
            number of rigid walls
        """

        # there have been cases that there are less than in the specs
        # indicated global vars. That breaks this computation, thus we
        # use max at the end.
        previous_global_vars = 6 + 7 * self._n_parts
        n_rigid_wall_vars = self.header.n_rigid_wall_vars
        n_rigid_walls = (self.header.n_global_vars - previous_global_vars) // n_rigid_wall_vars

        # if n_rigid_walls < 0:
        #     err_msg = "The computed number of rigid walls is negative ('{0}')."
        #     raise RuntimeError(err_msg.format(n_rigid_walls))

        return max(n_rigid_walls, 0)

    # pylint: disable = unused-argument, too-many-locals
    def _read_d3plot_file_generator(
        self, buffered_reading: bool, state_filter: Union[None, Set[int]]
    ) -> typing.Any:
        """Generator function for reading bare d3plot files

        Parameters
        ----------
        buffered_reading: bool
            whether to read one state at a time
        state_filter: Union[None, Set[int]]
            which states to filter out

        Yields
        ------
        buffer: BinaryBuffer
            buffer for each file
        n_states: int
            number of states from second yield on
        """

        # (1) STATES
        # This is dangerous. The following routine requires data from
        # several sections in the geometry part calling this too early crashes
        bytes_per_state = self._compute_n_bytes_per_state()
        file_infos = self._collect_file_infos(bytes_per_state)

        # some status
        n_files = len(file_infos)
        n_states = sum(map(lambda file_info: file_info.n_states, file_infos))

        # convert negative state indexes into positive ones
        if state_filter is not None:
            state_filter = _negative_to_positive_state_indexes(state_filter, n_states)

        # if using buffered reading, we load one state at a time
        # into memory
        if buffered_reading:
            file_infos_tmp: List[MemoryInfo] = []
            n_previous_states = 0
            for minfo in file_infos:
                for i_file_state in range(minfo.n_states):
                    i_global_state = n_previous_states + i_file_state

                    # do we need to skip this one
                    if state_filter and i_global_state not in state_filter:
                        continue

                    file_infos_tmp.append(
                        MemoryInfo(
                            start=minfo.start + i_file_state * bytes_per_state,
                            length=bytes_per_state,
                            filepath=minfo.filepath,
                            n_states=1,
                            filesize=minfo.filesize,
                            use_mmap=minfo.n_states != 1,
                        )
                    )

                n_previous_states += minfo.n_states
            file_infos = file_infos_tmp

        # number of states and if buffered reading is used
        n_states_selected = sum(map(lambda file_info: file_info.n_states, file_infos))
        yield n_states_selected

        sub_file_infos = [file_infos] if not buffered_reading else [[info] for info in file_infos]
        for sub_file_info_list in sub_file_infos:
            buffer, n_states = D3plot._read_file_from_memory_info(sub_file_info_list)
            yield buffer, n_states

    def _read_header(self):
        """Read the d3plot header"""

        if self._buffer:
            self._header.load_file(self._buffer)

        self.geometry_section_size = self._header.n_header_bytes

    def _parse_geometry(self):
        """Read the d3plot geometry"""

        # read material section
        self._read_material_section()

        # read fluid material data
        self._read_fluid_material_data()

        # SPH element data flags
        self._read_sph_element_data_flags()

        # Particle Data
        self._read_particle_data()

        # Geometry Data
        self._read_geometry_data()

        # User Material, Node, Blabla IDs
        self._read_user_ids()

        # Rigid Body Description
        self._read_rigid_body_description()

        # Adapted Element Parent List
        # manual says not implemented

        # Smooth Particle Hydrodynamcis Node and Material list
        self._read_sph_node_and_material_list()

        # Particle Geometry Data
        self._read_particle_geometry_data()

        # Rigid Road Surface Data
        self._read_rigid_road_surface()

        # Connectivity for weirdo elements
        # 10 Node Tetra
        # 8 Node Shell
        # 20 Node Solid
        # 27 Node Solid
        self._read_extra_node_connectivity()

        # Header Part & Contact Interface Titles
        # this is a class method since it is also needed elsewhere
        self.geometry_section_size = self._read_header_part_contact_interface_titles(
            self.header,
            self._buffer,
            self.geometry_section_size,  # type: ignore
            self.arrays,
        )

        # Extra Data Types (for multi solver output)
        # ... not supported

    def _read_material_section(self):
        """This function reads the material type section"""

        if not self._buffer:
            return

        if not self.header.has_material_type_section:
            return

        position = self.geometry_section_size

        # failsafe
        original_position = self.geometry_section_size
        blocksize = (2 + self.header.n_parts) * self.header.wordsize

        try:

            # Material Type Data
            #
            # "This data is required because those shell elements
            # that are in a rigid body have no element data output
            # in the state data section."
            #
            # "The normal length of the shell element state data is:
            # NEL4 * NV2D, when the MATTYP flag is set the length is:
            # (NEL4 – NUMRBE) * NV2D. When reading the shell element data,
            # the material number must be checked against IRBRTYP list to
            # find the element’s material type. If the type = 20, then
            # all the values for the element to zero." (Manual 03.2016)

            self._material_section_info.n_rigid_shells = int(
                self._buffer.read_number(position, self._header.itype)
            )  # type: ignore
            position += self.header.wordsize

            test_nummat = self._buffer.read_number(position, self._header.itype)
            position += self.header.wordsize

            if test_nummat != self.header.n_parts:
                raise RuntimeError(
                    "nmmat (header) != nmmat (material type data): "
                    f"{self.header.n_parts} != {test_nummat}",
                )

            self.arrays[ArrayType.part_material_type] = self._buffer.read_ndarray(
                position, self.header.n_parts * self.header.wordsize, 1, self.header.itype
            )
            position += self.header.n_parts * self.header.wordsize

        except Exception:
            # print info
            trb_msg = traceback.format_exc()
            # fix position
            position = original_position + blocksize

        self.geometry_section_size = position

    def _read_fluid_material_data(self):
        """Read the fluid material data"""

        if not self._buffer:
            return

        if self.header.n_ale_materials == 0:
            return

        position = self.geometry_section_size

        # safety
        original_position = position
        blocksize = self.header.n_ale_materials * self.header.wordsize

        try:
            # Fluid Material Data
            array_length = self.header.n_ale_materials * self.header.wordsize
            self.arrays[ArrayType.ale_material_ids] = self._buffer.read_ndarray(
                position, array_length, 1, self.header.itype
            )  # type: ignore
            position += array_length

        except Exception:

            # print info
            trb_msg = traceback.format_exc()
            msg = "A failure in %s was caught:\n%s"

            # fix position
            position = original_position + blocksize

        # remember position
        self.geometry_section_size = position

    def _read_sph_element_data_flags(self):
        """Read the sph element data flags"""

        if not self._buffer:
            return

        if not self.header.n_sph_nodes:
            return

        position = self.geometry_section_size

        sph_element_data_words = {
            "isphfg1": (position, self._header.itype),
            "isphfg2": (position + 1 * self.header.wordsize, self._header.itype),
            "isphfg3": (position + 2 * self.header.wordsize, self._header.itype),
            "isphfg4": (position + 3 * self.header.wordsize, self._header.itype),
            "isphfg5": (position + 4 * self.header.wordsize, self._header.itype),
            "isphfg6": (position + 5 * self.header.wordsize, self._header.itype),
            "isphfg7": (position + 6 * self.header.wordsize, self._header.itype),
            "isphfg8": (position + 7 * self.header.wordsize, self._header.itype),
            "isphfg9": (position + 8 * self.header.wordsize, self._header.itype),
            "isphfg10": (position + 9 * self.header.wordsize, self._header.itype),
            "isphfg11": (position + 10 * self.header.wordsize, self._header.itype),
        }

        sph_header_data = self.header.read_words(self._buffer, sph_element_data_words)

        self._sph_info.n_sph_array_length = sph_header_data["isphfg1"]
        self._sph_info.has_influence_radius = sph_header_data["isphfg2"] != 0
        self._sph_info.has_particle_pressure = sph_header_data["isphfg3"] != 0
        self._sph_info.has_stresses = sph_header_data["isphfg4"] != 0
        self._sph_info.has_plastic_strain = sph_header_data["isphfg5"] != 0
        self._sph_info.has_material_density = sph_header_data["isphfg6"] != 0
        self._sph_info.has_internal_energy = sph_header_data["isphfg7"] != 0
        self._sph_info.has_n_affecting_neighbors = sph_header_data["isphfg8"] != 0
        self._sph_info.has_strain_and_strainrate = sph_header_data["isphfg9"] != 0
        self._sph_info.has_true_strains = sph_header_data["isphfg9"] < 0
        self._sph_info.has_mass = sph_header_data["isphfg10"] != 0
        self._sph_info.n_sph_history_vars = sph_header_data["isphfg11"]

        if self._sph_info.n_sph_array_length != 11:
            msg = (
                "Detected inconsistency: "
                f"isphfg = {self._sph_info.n_sph_array_length} but must be 11."
            )
            raise RuntimeError(msg)

        self._sph_info.n_sph_vars = (
            sph_header_data["isphfg2"]
            + sph_header_data["isphfg3"]
            + sph_header_data["isphfg4"]
            + sph_header_data["isphfg5"]
            + sph_header_data["isphfg6"]
            + sph_header_data["isphfg7"]
            + sph_header_data["isphfg8"]
            + abs(sph_header_data["isphfg9"])
            + sph_header_data["isphfg10"]
            + sph_header_data["isphfg11"]
            + 1
        )  # material number

        self.geometry_section_size += sph_header_data["isphfg1"] * self.header.wordsize

    def _read_particle_data(self):
        """Read the geometry section for particle data (airbags)"""

        if not self._buffer:
            return

        if "npefg" not in self.header.raw_header:
            return
        npefg = self.header.raw_header["npefg"]

        # let's stick to the manual, too lazy to decypther this test
        if npefg <= 0 or npefg > 10000000:
            return

        position = self.geometry_section_size

        airbag_header = {
            # number of airbags
            "npartgas": npefg % 1000,
            # ?
            "subver": npefg // 1000,
        }

        particle_geometry_data_words = {
            # number of geometry variables
            "ngeom": (position, self._header.itype),
            # number of state variables
            "nvar": (position + 1 * self.header.wordsize, self._header.itype),
            # number of particles
            "npart": (position + 2 * self.header.wordsize, self._header.itype),
            # number of state geometry variables
            "nstgeom": (position + 3 * self.header.wordsize, self._header.itype),
        }

        self.header.read_words(self._buffer, particle_geometry_data_words, airbag_header)
        position += 4 * self.header.wordsize

        # transfer to info object
        self._airbag_info.n_airbags = npefg % 1000
        self._airbag_info.subver = npefg // 1000
        self._airbag_info.n_geometric_variables = airbag_header["ngeom"]
        self._airbag_info.n_particle_state_variables = airbag_header["nvar"]
        self._airbag_info.n_particles = airbag_header["npart"]
        self._airbag_info.n_airbag_state_variables = airbag_header["nstgeom"]

        if self._airbag_info.subver == 4:
            # number of chambers
            self._airbag_info.n_chambers = self._buffer.read_number(position, self._header.itype)
            position += self.header.wordsize

        n_airbag_variables = self._airbag_info.get_n_variables()

        # safety
        # from here on the code may fail
        original_position = position
        blocksize = 9 * n_airbag_variables * self.header.wordsize

        try:
            # variable typecodes
            self.arrays[ArrayType.airbag_variable_types] = self._buffer.read_ndarray(
                position, n_airbag_variables * self.header.wordsize, 1, self._header.itype
            )
            position += n_airbag_variables * self.header.wordsize

            # airbag variable names
            # every word is an ascii char
            airbag_variable_names = []
            var_width = 8

            for i_variable in range(n_airbag_variables):
                name = self._buffer.read_text(
                    position + (i_variable * var_width) * self.header.wordsize,
                    var_width * self.header.wordsize,
                )
                airbag_variable_names.append(name[:: self.header.wordsize])

            self.arrays[ArrayType.airbag_variable_names] = airbag_variable_names
            position += n_airbag_variables * var_width * self.header.wordsize

        except Exception:

            # print info
            trb_msg = traceback.format_exc()
            msg = "A failure in %s was caught:\n%s"

            # fix position
            position = original_position + blocksize

        # update position marker
        self.geometry_section_size = position

    # pylint: disable = too-many-branches
    def _read_geometry_data(self):
        """Read the data from the geometry section"""

        if not self._buffer:
            return

        # not sure but I think never used by LS-Dyna
        # anyway needs to be detected in the header and not here,
        # though it is mentioned in this section of the database manual
        #
        # is_packed = True if self.header['ndim'] == 3 else False
        # if is_packed:
        #     raise RuntimeError("Can not deal with packed "\
        #                        "geometry data (ndim == {}).".format(self.header['ndim']))

        position = self.geometry_section_size

        # node coords
        n_nodes = self.header.n_nodes
        n_dimensions = self.header.n_dimensions
        section_word_length = n_dimensions * n_nodes
        try:
            node_coordinates = self._buffer.read_ndarray(
                position, section_word_length * self.header.wordsize, 1, self.header.ftype
            ).reshape((n_nodes, n_dimensions))
            self.arrays[ArrayType.node_coordinates] = node_coordinates
        except Exception:
            trb_msg = traceback.format_exc()
            msg = "A failure in %d was caught:\n%s"
        finally:
            position += section_word_length * self.header.wordsize

        # solid data
        n_solids = self.header.n_solids
        section_word_length = 9 * n_solids
        try:
            elem_solid_data = self._buffer.read_ndarray(
                position, section_word_length * self.header.wordsize, 1, self._header.itype
            ).reshape((n_solids, 9))
            solid_connectivity = elem_solid_data[:, :8]
            solid_part_indexes = elem_solid_data[:, 8]
            self.arrays[ArrayType.element_solid_node_indexes] = solid_connectivity - FORTRAN_OFFSET
            self.arrays[ArrayType.element_solid_part_indexes] = solid_part_indexes - FORTRAN_OFFSET
        except Exception:
            trb_msg = traceback.format_exc()
            msg = "A failure in %s was caught:\n%s"
        finally:
            position += section_word_length * self.header.wordsize

        # ten node solids extra nodes
        if self.header.has_solid_2_extra_nodes:
            section_word_length = 2 * n_solids
            try:
                self.arrays[
                    ArrayType.element_solid_extra_nodes
                ] = elem_solid_data = self._buffer.read_ndarray(
                    position, section_word_length * self.header.wordsize, 1, self._header.itype
                ).reshape(
                    (n_solids, 2)
                )
            except Exception:
                trb_msg = traceback.format_exc()
                msg = "A failure in %s was caught:\n%s"
            finally:
                position += section_word_length * self.header.wordsize

        # 8 node thick shells
        n_thick_shells = self.header.n_thick_shells
        section_word_length = 9 * n_thick_shells
        try:
            elem_tshell_data = self._buffer.read_ndarray(
                position, section_word_length * self.header.wordsize, 1, self._header.itype
            ).reshape((self.header.n_thick_shells, 9))
            self.arrays[ArrayType.element_tshell_node_indexes] = (
                elem_tshell_data[:, :8] - FORTRAN_OFFSET
            )
            self.arrays[ArrayType.element_tshell_part_indexes] = (
                elem_tshell_data[:, 8] - FORTRAN_OFFSET
            )
        except Exception:
            trb_msg = traceback.format_exc()
            msg = "A failure in %s was caught:\n%s"
        finally:
            position += section_word_length * self.header.wordsize

        # beams
        n_beams = self.header.n_beams
        section_word_length = 6 * n_beams
        try:
            elem_beam_data = self._buffer.read_ndarray(
                position, section_word_length * self.header.wordsize, 1, self._header.itype
            ).reshape((n_beams, 6))
            self.arrays[ArrayType.element_beam_part_indexes] = elem_beam_data[:, 5] - FORTRAN_OFFSET
            self.arrays[ArrayType.element_beam_node_indexes] = (
                elem_beam_data[:, :5] - FORTRAN_OFFSET
            )
        except Exception:
            trb_msg = traceback.format_exc()
            msg = "A failure in %s was caught:\n%s"
        finally:
            position += section_word_length * self.header.wordsize

        # shells
        n_shells = self.header.n_shells
        section_word_length = 5 * n_shells
        try:
            elem_shell_data = self._buffer.read_ndarray(
                position, section_word_length * self.header.wordsize, 1, self._header.itype
            ).reshape((self.header.n_shells, 5))
            self.arrays[ArrayType.element_shell_node_indexes] = (
                elem_shell_data[:, :4] - FORTRAN_OFFSET
            )
            self.arrays[ArrayType.element_shell_part_indexes] = (
                elem_shell_data[:, 4] - FORTRAN_OFFSET
            )
        except Exception:
            trb_msg = traceback.format_exc()
            msg = "A failure in %s was caught:\n%s"
        finally:
            position += section_word_length * self.header.wordsize

        # update word position
        self.geometry_section_size = position

    def _read_user_ids(self):

        if not self._buffer:
            return

        if not self.header.has_numbering_section:
            self.arrays[ArrayType.node_ids] = np.arange(
                FORTRAN_OFFSET, self.header.n_nodes + FORTRAN_OFFSET, dtype=self.header.itype
            )
            self.arrays[ArrayType.element_solid_ids] = np.arange(
                FORTRAN_OFFSET, self.header.n_solids + FORTRAN_OFFSET, dtype=self.header.itype
            )
            self.arrays[ArrayType.element_beam_ids] = np.arange(
                FORTRAN_OFFSET, self.header.n_beams + FORTRAN_OFFSET, dtype=self.header.itype
            )
            self.arrays[ArrayType.element_shell_ids] = np.arange(
                FORTRAN_OFFSET, self.header.n_shells + FORTRAN_OFFSET, dtype=self.header.itype
            )
            self.arrays[ArrayType.element_tshell_ids] = np.arange(
                FORTRAN_OFFSET, self.header.n_thick_shells + FORTRAN_OFFSET, dtype=self.header.itype
            )
            self.arrays[ArrayType.part_ids] = np.arange(
                FORTRAN_OFFSET, self.header.n_parts + FORTRAN_OFFSET, dtype=self.header.itype
            )
            return

        position = self.geometry_section_size

        # safety
        original_position = position
        blocksize = self.header.raw_header["narbs"] * self.header.wordsize

        try:
            numbering_words = {
                "nsort": (position, self._header.itype),
                "nsrh": (position + 1 * self.header.wordsize, self._header.itype),
                "nsrb": (position + 2 * self.header.wordsize, self._header.itype),
                "nsrs": (position + 3 * self.header.wordsize, self._header.itype),
                "nsrt": (position + 4 * self.header.wordsize, self._header.itype),
                "nsortd": (position + 5 * self.header.wordsize, self._header.itype),
                "nsrhd": (position + 6 * self.header.wordsize, self._header.itype),
                "nsrbd": (position + 7 * self.header.wordsize, self._header.itype),
                "nsrsd": (position + 8 * self.header.wordsize, self._header.itype),
                "nsrtd": (position + 9 * self.header.wordsize, self._header.itype),
            }

            extra_numbering_words = {
                "nsrma": (position + 10 * self.header.wordsize, self._header.itype),
                "nsrmu": (position + 11 * self.header.wordsize, self._header.itype),
                "nsrmp": (position + 12 * self.header.wordsize, self._header.itype),
                "nsrtm": (position + 13 * self.header.wordsize, self._header.itype),
                "numrbs": (position + 14 * self.header.wordsize, self._header.itype),
                "nmmat": (position + 15 * self.header.wordsize, self._header.itype),
            }

            numbering_header = self.header.read_words(self._buffer, numbering_words)
            position += len(numbering_words) * self.header.wordsize

            # let's make life easier
            info = self._numbering_info

            # transfer first bunch
            info.ptr_node_ids = abs(numbering_header["nsort"])
            info.has_material_ids = numbering_header["nsort"] < 0
            info.ptr_solid_ids = numbering_header["nsrh"]
            info.ptr_beam_ids = numbering_header["nsrb"]
            info.ptr_shell_ids = numbering_header["nsrs"]
            info.ptr_thick_shell_ids = numbering_header["nsrt"]
            info.n_nodes = numbering_header["nsortd"]
            info.n_solids = numbering_header["nsrhd"]
            info.n_beams = numbering_header["nsrbd"]
            info.n_shells = numbering_header["nsrsd"]
            info.n_thick_shells = numbering_header["nsrtd"]

            if info.has_material_ids:

                # read extra header
                self.header.read_words(self._buffer, extra_numbering_words, numbering_header)
                position += len(extra_numbering_words) * self.header.wordsize

                # transfer more
                info.ptr_material_ids = numbering_header["nsrma"]
                info.ptr_material_ids_defined_order = numbering_header["nsrmu"]
                info.ptr_material_ids_crossref = numbering_header["nsrmp"]
                info.n_parts = numbering_header["nsrtm"]
                info.n_rigid_bodies = numbering_header["numrbs"]
                info.n_parts2 = numbering_header["nmmat"]
            else:
                info.n_parts = self.header.n_parts

            # let's do a quick check
            n_words_computed = (
                len(numbering_header)
                + info.n_nodes
                + info.n_shells
                + info.n_beams
                + info.n_solids
                + info.n_thick_shells
                + info.n_parts * 3
            )
            if n_words_computed != self.header.n_numbering_section_words:
                warn_msg = (
                    "ID section: The computed word count does "
                    "not match the header word count: %d != %d."
                    " The ID arrays might contain errors."
                )
            # node ids
            array_length = info.n_nodes * self.header.wordsize
            self.arrays[ArrayType.node_ids] = self._buffer.read_ndarray(
                position, array_length, 1, self._header.itype
            )
            position += array_length
            # solid ids
            array_length = info.n_solids * self.header.wordsize
            self.arrays[ArrayType.element_solid_ids] = self._buffer.read_ndarray(
                position, array_length, 1, self._header.itype
            )
            position += array_length
            # beam ids
            array_length = info.n_beams * self.header.wordsize
            self.arrays[ArrayType.element_beam_ids] = self._buffer.read_ndarray(
                position, array_length, 1, self._header.itype
            )
            position += array_length
            # shell ids
            array_length = info.n_shells * self.header.wordsize
            self.arrays[ArrayType.element_shell_ids] = self._buffer.read_ndarray(
                position, array_length, 1, self._header.itype
            )
            position += array_length
            # tshell ids
            array_length = info.n_thick_shells * self.header.wordsize
            self.arrays[ArrayType.element_tshell_ids] = self._buffer.read_ndarray(
                position, array_length, 1, self._header.itype
            )
            position += array_length

            # part ids
            #
            # this makes no sense but materials are output three times at this section
            # but the length of the array (nmmat) is only output if nsort < 0. In
            # the other case the length is unknown ...
            #
            # Bugfix:
            # The material arrays (three times) are always output, even if nsort < 0
            # which means they are not used. Quite confusing, especially since nmmat
            # is output in the main header and numbering header.
            #
            if "nmmat" in numbering_header:

                if info.n_parts != self.header.n_parts:
                    err_msg = (
                        "nmmat in the file header (%d) and in the "
                        "numbering header (%d) are inconsistent."
                    )
                    raise RuntimeError(err_msg, self.header.n_parts, info.n_parts)

                array_length = info.n_parts * self.header.wordsize

                self.arrays[ArrayType.part_ids] = self._buffer.read_ndarray(
                    position, info.n_parts * self.header.wordsize, 1, self._header.itype
                )
                position += info.n_parts * self.header.wordsize

                self.arrays[ArrayType.part_ids_unordered] = self._buffer.read_ndarray(
                    position, info.n_parts * self.header.wordsize, 1, self._header.itype
                )
                position += info.n_parts * self.header.wordsize

                self.arrays[ArrayType.part_ids_cross_references] = self._buffer.read_ndarray(
                    position, info.n_parts * self.header.wordsize, 1, self._header.itype
                )
                position += info.n_parts * self.header.wordsize

            else:
                position += 3 * self.header.n_parts * self.header.wordsize

        except Exception:
            # print info
            trb_msg = traceback.format_exc()
            msg = "A failure in %s was caught:\n%s"

            # fix position
            position = original_position + blocksize

        # update position
        self.geometry_section_size = position

    def _read_rigid_body_description(self):
        """Read the rigid body description section"""

        if not self._buffer:
            return

        if not self.header.has_rigid_body_data:
            return

        position = self.geometry_section_size

        rigid_body_description_header = {
            "nrigid": self._buffer.read_number(position, self._header.itype)
        }
        position += self.header.wordsize

        info = self._rigid_body_info
        info.n_rigid_bodies = rigid_body_description_header["nrigid"]

        rigid_bodies: List[RigidBodyMetadata] = []
        for _ in range(info.n_rigid_bodies):
            rigid_body_info = {
                # rigid body part internal number
                "mrigid": self._buffer.read_number(position, self._header.itype),
                # number of nodes in rigid body
                "numnodr": self._buffer.read_number(
                    position + self.header.wordsize, self._header.itype
                ),
            }
            position += 2 * self.header.wordsize

            # internal node number of rigid body
            array_length = rigid_body_info["numnodr"] * self.header.wordsize
            rigid_body_info["noder"] = self._buffer.read_ndarray(
                position, array_length, 1, self._header.itype
            )
            position += array_length

            # number of active (non-rigid) nodes
            rigid_body_info["numnoda"] = self._buffer.read_number(position, self._header.itype)
            position += self.header.wordsize

            # internal node numbers of active nodes
            array_length = rigid_body_info["numnoda"] * self.header.wordsize
            rigid_body_info["nodea"] = self._buffer.read_ndarray(
                position, array_length, 1, self._header.itype
            )
            position += array_length

            # transfer props
            body_metadata = RigidBodyMetadata(
                internal_number=rigid_body_info["mrigid"],
                n_nodes=rigid_body_info["numnodr"],
                node_indexes=rigid_body_info["noder"],
                n_active_nodes=rigid_body_info["numnoda"],
                active_node_indexes=rigid_body_info["nodea"],
            )

            # append to list
            rigid_bodies.append(body_metadata)

        # save rigid body info to header
        info.rigid_body_metadata_list = rigid_bodies

        # save arrays
        rigid_body_n_nodes = []
        rigid_body_part_indexes = []
        rigid_body_n_active_nodes = []
        rigid_body_node_indexes_list = []
        rigid_body_active_node_indexes_list = []
        for rigid_body_info in rigid_bodies:
            rigid_body_part_indexes.append(rigid_body_info.internal_number)
            rigid_body_n_nodes.append(rigid_body_info.n_nodes)
            rigid_body_node_indexes_list.append(rigid_body_info.node_indexes - FORTRAN_OFFSET)
            rigid_body_n_active_nodes.append(rigid_body_info.n_active_nodes)
            rigid_body_active_node_indexes_list.append(
                rigid_body_info.active_node_indexes - FORTRAN_OFFSET
            )

        self.arrays[ArrayType.rigid_body_part_indexes] = (
            np.array(rigid_body_part_indexes, dtype=self._header.itype) - FORTRAN_OFFSET
        )
        self.arrays[ArrayType.rigid_body_n_nodes] = np.array(
            rigid_body_n_nodes, dtype=self._header.itype
        )
        self.arrays[ArrayType.rigid_body_n_active_nodes] = np.array(
            rigid_body_n_active_nodes, dtype=self._header.itype
        )
        self.arrays[ArrayType.rigid_body_node_indexes_list] = rigid_body_node_indexes_list
        self.arrays[
            ArrayType.rigid_body_active_node_indexes_list
        ] = rigid_body_active_node_indexes_list

        # update position
        self.geometry_section_size = position

    def _read_sph_node_and_material_list(self):
        """Read SPH node and material list"""

        if not self._buffer:
            return

        if self.header.n_sph_nodes <= 0:
            return

        position = self.geometry_section_size

        array_length = self.header.n_sph_nodes * self.header.wordsize * 2
        try:
            # read info array
            sph_node_matlist = self._buffer.read_ndarray(
                position, array_length, 1, self._header.itype
            ).reshape((self.header.n_sph_nodes, 2))

            # save array
            self.arrays[ArrayType.sph_node_indexes] = sph_node_matlist[:, 0] - FORTRAN_OFFSET
            self.arrays[ArrayType.sph_node_material_index] = sph_node_matlist[:, 1] - FORTRAN_OFFSET

        except Exception:
            # print info
            trb_msg = traceback.format_exc()
            msg = "A failure in %s was caught:\n%s"

        finally:
            # update position
            self.geometry_section_size += array_length

    def _read_particle_geometry_data(self):
        """Read the particle geometry data"""

        if not self._buffer:
            return

        if "npefg" not in self.header.raw_header:
            return

        if self.header.raw_header["npefg"] <= 0:
            return

        info = self._airbag_info

        position = self.geometry_section_size

        # size of geometry section checking
        ngeom = info.n_geometric_variables
        if ngeom not in [4, 5]:
            raise RuntimeError("variable ngeom in the airbag header must be 4 or 5.")

        original_position = position
        blocksize = info.n_airbags * ngeom * self.header.wordsize
        try:

            # extract geometry as a single array
            array_length = blocksize
            particle_geom_data = self._buffer.read_ndarray(
                position, array_length, 1, self._header.itype
            ).reshape((info.n_airbags, ngeom))
            position += array_length

            # store arrays
            self.arrays[ArrayType.airbags_first_particle_id] = particle_geom_data[:, 0]
            self.arrays[ArrayType.airbags_n_particles] = particle_geom_data[:, 1]
            self.arrays[ArrayType.airbags_ids] = particle_geom_data[:, 2]
            self.arrays[ArrayType.airbags_n_gas_mixtures] = particle_geom_data[:, 3]
            if ngeom == 5:
                self.arrays[ArrayType.airbags_n_chambers] = particle_geom_data[:, 4]

        except Exception:
            # print info
            trb_msg = traceback.format_exc()
            msg = "A failure in %d was caught:\n%s"

            # fix position
            position = original_position + blocksize

        # update position
        self.geometry_section_size = position


    def _read_rigid_road_surface(self):
        """Read rigid road surface data"""

        if not self._buffer:
            return

        if not self.header.has_rigid_road_surface:
            return

        position = self.geometry_section_size

        # read header
        rigid_road_surface_words = {
            "nnode": (position, self._header.itype),
            "nseg": (position + 1 * self.header.wordsize, self._header.itype),
            "nsurf": (position + 2 * self.header.wordsize, self._header.itype),
            "motion": (position + 3 * self.header.wordsize, self._header.itype),
        }

        rigid_road_header = self.header.read_words(self._buffer, rigid_road_surface_words)
        position += 4 * self.header.wordsize

        self._rigid_road_info = RigidRoadInfo(
            n_nodes=rigid_road_header["nnode"],
            n_roads=rigid_road_header["nsurf"],
            n_road_segments=rigid_road_header["nseg"],
            motion=rigid_road_header["motion"],
        )
        info = self._rigid_road_info

        # node ids
        array_length = info.n_nodes * self.header.wordsize
        rigid_road_node_ids = self._buffer.read_ndarray(
            position, array_length, 1, self._header.itype
        )
        self.arrays[ArrayType.rigid_road_node_ids] = rigid_road_node_ids
        position += array_length

        # node xyz
        array_length = info.n_nodes * 3 * self.header.wordsize
        rigid_road_node_coords = self._buffer.read_ndarray(
            position, array_length, 1, self.header.ftype
        ).reshape((info.n_nodes, 3))
        self.arrays[ArrayType.rigid_road_node_coordinates] = rigid_road_node_coords
        position += array_length

        # read road segments
        # Warning: must be copied
        rigid_road_ids = np.empty(info.n_roads, dtype=self._header.itype)
        rigid_road_nsegments = np.empty(info.n_roads, dtype=self._header.itype)
        rigid_road_segment_node_ids = []

        # this array is created since the array database requires
        # constant sized arrays, and we dump all segments into one
        # array. In order to distinguish which segment
        # belongs to which road, this new array keeps track of it
        rigid_road_segment_road_id = []

        # n_total_segments = 0
        for i_surf in range(info.n_roads):
            # surface id
            surf_id = self._buffer.read_number(position, self._header.itype)  # type: ignore
            position += self.header.wordsize
            rigid_road_ids[i_surf] = surf_id

            # number of segments of surface
            surf_nseg = self._buffer.read_number(
                position + 1 * self.header.wordsize, self._header.itype
            )  # type: ignore
            position += self.header.wordsize
            rigid_road_nsegments[i_surf] = surf_nseg

            # count total segments
            # n_total_segments += surf_nseg

            # node ids of surface segments
            array_length = 4 * surf_nseg * self.header.wordsize
            surf_segm_node_ids = self._buffer.read_ndarray(
                position,  # type: ignore
                array_length,  # type: ignore
                1,
                self._header.itype,
            ).reshape((surf_nseg, 4))
            position += array_length
            rigid_road_segment_node_ids.append(surf_segm_node_ids)

            # remember road id for segments
            rigid_road_segment_road_id += [surf_id] * surf_nseg

        # save arrays
        self.arrays[ArrayType.rigid_road_ids] = rigid_road_ids
        self.arrays[ArrayType.rigid_road_n_segments] = rigid_road_nsegments
        self.arrays[ArrayType.rigid_road_segment_node_ids] = np.concatenate(
            rigid_road_segment_node_ids
        )
        self.arrays[ArrayType.rigid_road_segment_road_id] = np.asarray(rigid_road_segment_road_id)

        # update position
        self.geometry_section_size = position

    # pylint: disable = too-many-branches
    def _read_extra_node_connectivity(self):
        """Read the extra node data for creepy elements"""

        if not self._buffer:
            return

        position = self.geometry_section_size

        # extra 2 node connectivity for 10 node tetrahedron elements
        if self.header.has_solid_2_extra_nodes:
            array_length = 2 * self.header.n_solids * self.header.wordsize
            try:
                array = self._buffer.read_ndarray(
                    position, array_length, 1, self._header.itype
                ).reshape((self.header.n_solids, 2))
                self.arrays[ArrayType.element_solid_node10_extra_node_indexes] = (
                    array - FORTRAN_OFFSET
                )
            except Exception:
                trb_msg = traceback.format_exc()
                msg = "A failure in %s was caught:\n%s"
            finally:
                position += array_length

        # 8 node shell elements
        if self.header.n_shells_8_nodes > 0:
            array_length = 5 * self.header.n_shells_8_nodes * self.header.wordsize
            try:
                array = self._buffer.read_ndarray(
                    position, array_length, 1, self._header.itype
                ).reshape((self.header.n_shells_8_nodes, 5))
                self.arrays[ArrayType.element_shell_node8_element_index] = (
                    array[:, 0] - FORTRAN_OFFSET
                )
                self.arrays[ArrayType.element_shell_node8_extra_node_indexes] = (
                    array[:, 1:] - FORTRAN_OFFSET
                )
            except Exception:
                trb_msg = traceback.format_exc()
                msg = "A failure in %s was caught:\n%s"
            finally:
                position += array_length

        # 20 node solid elements
        if self.header.n_solids_20_node_hexas > 0:
            array_length = 13 * self.header.n_solids_20_node_hexas * self.header.wordsize
            try:
                array = self._buffer.read_ndarray(
                    position, array_length, 1, self._header.itype
                ).reshape((self.header.n_solids_20_node_hexas, 13))
                self.arrays[ArrayType.element_solid_node20_element_index] = (
                    array[:, 0] - FORTRAN_OFFSET
                )
                self.arrays[ArrayType.element_solid_node20_extra_node_indexes] = (
                    array[:, 1:] - FORTRAN_OFFSET
                )
            except Exception:
                trb_msg = traceback.format_exc()
                msg = "A failure in %s was caught:\n%s"
            finally:
                position += array_length

        # 27 node solid hexas
        if (
            self.header.n_solids_27_node_hexas > 0
            and self.header.quadratic_elems_has_full_connectivity
        ):
            array_length = 28 * self.header.n_solids_27_node_hexas * self.header.wordsize
            try:
                array = self._buffer.read_ndarray(
                    position, array_length, 1, self._header.itype
                ).reshape((self.header.n_solids_27_node_hexas, 28))
                self.arrays[ArrayType.element_solid_node27_element_index] = (
                    array[:, 0] - FORTRAN_OFFSET
                )
                self.arrays[ArrayType.element_solid_node27_extra_node_indexes] = (
                    array[:, 1:] - FORTRAN_OFFSET
                )
            except Exception:
                trb_msg = traceback.format_exc()
                msg = "A failure in %s was caught:\n%s"
            finally:
                position += array_length

        # 21 node solid pentas
        if (
            self.header.n_solids_21_node_pentas > 0
            and self.header.quadratic_elems_has_full_connectivity
        ):
            array_length = 22 * self.header.n_solids_21_node_pentas * self.header.wordsize
            try:
                array = self._buffer.read_ndarray(
                    position, array_length, 1, self._header.itype
                ).reshape((self.header.n_solids_21_node_pentas, 22))
                self.arrays[ArrayType.element_solid_node21_penta_element_index] = (
                    array[:, 0] - FORTRAN_OFFSET
                )
                self.arrays[ArrayType.element_solid_node21_penta_extra_node_indexes] = (
                    array[:, 1:] - FORTRAN_OFFSET
                )
            except Exception:
                trb_msg = traceback.format_exc()
                msg = "A failure in %s was caught:\n%s"
            finally:
                position += array_length

        # 15 node solid tetras
        if (
            self.header.n_solids_15_node_tetras > 0
            and self.header.quadratic_elems_has_full_connectivity
        ):
            # manual says 8 but this seems odd
            array_length = 8 * self.header.n_solids_15_node_tetras * self.header.wordsize
            try:
                array = self._buffer.read_ndarray(
                    position, array_length, 1, self._header.itype
                ).reshape((self.header.n_solids_15_node_tetras, 8))
                self.arrays[ArrayType.element_solid_node15_tetras_element_index] = (
                    array[:, 0] - FORTRAN_OFFSET
                )
                self.arrays[ArrayType.element_solid_node15_tetras_extra_node_indexes] = (
                    array[:, 1:] - FORTRAN_OFFSET
                )
            except Exception:
                trb_msg = traceback.format_exc()
                msg = "A failure in %s was caught:\n%s"
            finally:
                position += array_length

        # 20 node solid tetras
        if self.header.n_solids_20_node_tetras > 0 and self.header.has_cubic_solids:
            array_length = 21 * self.header.n_solids_20_node_tetras * self.header.wordsize
            try:
                array = self._buffer.read_ndarray(
                    position, array_length, 1, self._header.itype
                ).reshape((self.header.n_solids_20_node_tetras, 21))
                self.arrays[ArrayType.element_solid_node20_tetras_element_index] = (
                    array[:, 0] - FORTRAN_OFFSET
                )
                self.arrays[ArrayType.element_solid_node20_tetras_extra_node_indexes] = (
                    array[:, 1:] - FORTRAN_OFFSET
                )
            except Exception:
                trb_msg = traceback.format_exc()
                msg = "A failure in %s was caught:\n%s"
            finally:
                position += array_length

        # 40 node solid tetras
        if self.header.n_solids_40_node_pentas > 0 and self.header.has_cubic_solids:
            array_length = 41 * self.header.n_solids_40_node_pentas * self.header.wordsize
            try:
                array = self._buffer.read_ndarray(
                    position, array_length, 1, self._header.itype
                ).reshape((self.header.n_solids_40_node_pentas, 41))
                self.arrays[ArrayType.element_solid_node40_pentas_element_index] = (
                    array[:, 0] - FORTRAN_OFFSET
                )
                self.arrays[ArrayType.element_solid_node40_pentas_extra_node_indexes] = (
                    array[:, 1:] - FORTRAN_OFFSET
                )
            except Exception:
                trb_msg = traceback.format_exc()
                msg = "A failure in %s was caught:\n%s"
            finally:
                position += array_length

        # 64 node solid tetras
        if self.header.n_solids_64_node_hexas > 0 and self.header.has_cubic_solids:
            array_length = 65 * self.header.n_solids_64_node_hexas * self.header.wordsize
            try:
                array = self._buffer.read_ndarray(
                    position, array_length, 1, self._header.itype
                ).reshape((self.header.n_solids_64_node_hexas, 65))
                self.arrays[ArrayType.element_solid_node64_hexas_element_index] = (
                    array[:, 0] - FORTRAN_OFFSET
                )
                self.arrays[ArrayType.element_solid_node64_hexas_extra_node_indexes] = (
                    array[:, 1:] - FORTRAN_OFFSET
                )
            except Exception:
                trb_msg = traceback.format_exc()
                msg = "A failure in %s was caught:\n%s"
            finally:
                position += array_length

        # update position
        self.geometry_section_size = position

    # pylint: disable = too-many-branches
    @classmethod
    def _read_header_part_contact_interface_titles(
        cls,
        header: D3plotHeader,
        buffer: Union[BinaryBuffer, None],
        geometry_section_size: int,
        arrays: dict,
    ) -> int:
        """Read the header for the parts, contacts and interfaces

        Parameters
        ----------
        header: D3plotHeader
            d3plot header
        bb: BinaryBuffer
            buffer holding geometry
        geometry_section_size: int
            size of the geometry section until now
        arrays: dict
            dictionary holding arrays and where arrays will be saved into

        Returns
        -------
        geometry_section_size: int
            new size of the geometry section
        """

        if not buffer:
            return geometry_section_size

        if header.filetype not in (
            D3plotFiletype.D3PLOT,
            D3plotFiletype.D3PART,
            D3plotFiletype.INTFOR,
        ):
            return geometry_section_size


        position = geometry_section_size

        # Security
        #
        # we try to read the titles ahead. If dyna writes multiple files
        # then the first file is geometry only thus failing here has no
        # impact on further state reading.
        # If though states are compressed into the first file then we are
        # in trouble here even when catching here.
        try:
            # there is only output if there is an eof marker
            # at least I think I fixed such a bug in the past
            if not cls._is_end_of_file_marker(buffer, position, header.ftype):
                return geometry_section_size

            position += header.wordsize

            # section have types here according to what is inside
            ntypes = []

            # read first ntype
            current_ntype = buffer.read_number(position, header.itype)

            while current_ntype in [90000, 90001, 90002, 90020]:

                # title output
                if current_ntype == 90000:

                    ntypes.append(current_ntype)
                    position += header.wordsize

                    # Bugfix:
                    # the titles are always 18*4 bytes, even if the wordsize
                    # is 8 bytes for the entire file.
                    titles_wordsize = 4

                    array_length = 18 * titles_wordsize
                    header.title2 = buffer.read_text(position, array_length)
                    position += array_length

                # some title output
                elif current_ntype in [90001, 90002, 90020]:

                    ntypes.append(current_ntype)
                    position += header.wordsize

                    # number of parts
                    entry_count = buffer.read_number(position, header.itype)
                    position += header.wordsize

                    # Bugfix:
                    # the titles are always 18*4 bytes, even if the wordsize
                    # is 8 bytes for the entire file.
                    titles_wordsize = 4

                    # part ids and corresponding titles
                    array_type = np.dtype(
                        [("ids", header.itype), ("titles", "S" + str(18 * titles_wordsize))]
                    )
                    array_length = (header.wordsize + 18 * titles_wordsize) * int(entry_count)
                    tmp_arrays = buffer.read_ndarray(position, array_length, 1, array_type)
                    position += array_length

                    # save stuff
                    if current_ntype == 90001:
                        arrays[ArrayType.part_titles_ids] = tmp_arrays["ids"]
                        arrays[ArrayType.part_titles] = tmp_arrays["titles"]
                    elif current_ntype == 90002:
                        arrays[ArrayType.contact_title_ids] = tmp_arrays["ids"]
                        arrays[ArrayType.contact_titles] = tmp_arrays["titles"]
                    elif current_ntype == 90020:
                        arrays["icfd_part_title_ids"] = tmp_arrays["ids"]
                        arrays["icfd_part_titles"] = tmp_arrays["titles"]

                # d3prop
                elif current_ntype == 90100:

                    ntypes.append(current_ntype)
                    position += header.wordsize

                    # number of keywords
                    nline = buffer.read_number(position, header.itype)
                    position += header.wordsize

                    # Bugfix:
                    # the titles are always 18*4 bytes, even if the wordsize
                    # is 8 bytes for the entire file.
                    titles_wordsize = 4

                    # keywords
                    array_length = 20 * titles_wordsize * int(nline)
                    d3prop_keywords = buffer.read_ndarray(
                        position, array_length, 1, np.dtype("S" + str(titles_wordsize * 20))
                    )
                    position += array_length

                    # save
                    arrays["d3prop_keywords"] = d3prop_keywords

                # not sure whether there is an eof file here
                # do not have a test file to check ...
                if cls._is_end_of_file_marker(buffer, position, header.ftype):
                    position += header.wordsize

                # next one
                if buffer.size <= position:
                    break
                current_ntype = buffer.read_number(position, header.itype)

            header.n_types = tuple(ntypes)

        except Exception:
            trb_msg = traceback.format_exc()
            msg = "A failure in %s was caught:\n%s"

        # remember position
        geometry_section_size = position

        return geometry_section_size

    @staticmethod
    def _read_states_allocate_arrays(
        header: D3plotHeader,
        material_section_info: MaterialSectionInfo,
        airbag_info: AirbagInfo,
        rigid_road_info: RigidRoadInfo,
        rigid_body_info: RigidBodyInfo,
        n_states: int,
        n_rigid_walls: int,
        n_parts: int,
        array_names: Union[Iterable[str], None],
        array_dict: dict,
    ) -> None:
        """Allocate the state arrays

        Parameters
        ----------
        header: D3plotHeader
            header of the d3plot
        material_section_info: MaterialSectionInfo
            info about the material section data
        airbag_info: AirbagInfo
            info for airbags
        rigid_road_info: RigidRoadInfo
            info for rigid roads
        rigid_body_info: RigidBodyInfo
            info for rigid bodies
        n_states: int
            number of states to allocate memory for
        n_rigid_walls: int
            number of rigid walls
        n_parts: int
            number of parts
        array_names: Union[Iterable[str], None]
            names of state arrays to allocate (all if None)
        array_dict: dict
            dictionary to allocate arrays into
        """

        # (1) ARRAY SHAPES
        # general
        n_dim = header.n_dimensions
        # nodes
        n_nodes = header.n_nodes
        # solids
        n_solids = header.n_solids
        n_solids_thermal_vars = header.n_solid_thermal_vars
        n_solids_strain_vars = 6 * header.has_element_strain * (header.n_solid_history_vars >= 6)
        n_solid_thermal_strain_vars = 6 * header.has_solid_shell_thermal_strain_tensor
        n_solid_plastic_strain_vars = 6 * header.has_solid_shell_plastic_strain_tensor
        n_solid_layers = header.n_solid_layers
        n_solids_history_vars = (
            header.n_solid_history_vars
            - n_solids_strain_vars
            - n_solid_thermal_strain_vars
            - n_solid_plastic_strain_vars
        )
        # thick shells
        n_tshells = header.n_thick_shells
        n_tshells_history_vars = header.n_shell_tshell_history_vars
        n_tshells_layers = header.n_shell_tshell_layers
        # beams
        n_beams = header.n_beams
        n_beams_history_vars = header.n_beam_history_vars
        n_beam_vars = header.n_beam_vars
        n_beams_layers = max(
            int((-3 * n_beams_history_vars + n_beam_vars - 6) / (n_beams_history_vars + 5)), 0
        )
        # shells
        n_shells = header.n_shells
        n_shells_reduced = header.n_shells - material_section_info.n_rigid_shells
        n_shell_layers = header.n_shell_tshell_layers
        n_shell_history_vars = header.n_shell_tshell_history_vars
        # sph
        allocate_sph = header.n_sph_nodes != 0
        n_sph_particles = header.n_sph_nodes if allocate_sph else 0
        # airbags
        allocate_airbags = header.n_airbags != 0
        n_airbags = header.n_airbags if allocate_airbags else 0
        n_airbag_particles = airbag_info.n_particles if allocate_airbags else 0
        # rigid roads
        allocate_rigid_roads = rigid_road_info.n_roads != 0
        n_roads = rigid_road_info.n_roads if allocate_rigid_roads else 0
        # rigid bodies
        n_rigid_bodies = rigid_body_info.n_rigid_bodies

        # dictionary to lookup array types
        state_array_shapes = {
            # global
            ArrayType.global_timesteps: [n_states],
            ArrayType.global_kinetic_energy: [n_states],
            ArrayType.global_internal_energy: [n_states],
            ArrayType.global_total_energy: [n_states],
            ArrayType.global_velocity: [n_states, 3],
            # parts
            ArrayType.part_internal_energy: [n_states, n_parts],
            ArrayType.part_kinetic_energy: [n_states, n_parts],
            ArrayType.part_velocity: [n_states, n_parts, 3],
            ArrayType.part_mass: [n_states, n_parts],
            ArrayType.part_hourglass_energy: [n_states, n_parts],
            # rigid wall
            ArrayType.rigid_wall_force: [n_states, n_rigid_walls],
            ArrayType.rigid_wall_position: [n_states, n_rigid_walls, 3],
            # nodes
            ArrayType.node_temperature: [n_states, n_nodes, 3]
            if header.has_node_temperature_layers
            else [n_states, n_nodes],
            ArrayType.node_heat_flux: [n_states, n_nodes, 3],
            ArrayType.node_mass_scaling: [n_states, n_nodes],
            ArrayType.node_displacement: [n_states, n_nodes, n_dim],
            ArrayType.node_velocity: [n_states, n_nodes, n_dim],
            ArrayType.node_acceleration: [n_states, n_nodes, n_dim],
            ArrayType.node_temperature_gradient: [n_states, n_nodes],
            ArrayType.node_residual_forces: [n_states, n_nodes, 3],
            ArrayType.node_residual_moments: [n_states, n_nodes, 3],
            # solids
            ArrayType.element_solid_thermal_data: [n_states, n_solids, n_solids_thermal_vars],
            ArrayType.element_solid_stress: [n_states, n_solids, n_solid_layers, 6],
            ArrayType.element_solid_effective_plastic_strain: [n_states, n_solids, n_solid_layers],
            ArrayType.element_solid_history_variables: [
                n_states,
                n_solids,
                n_solid_layers,
                n_solids_history_vars,
            ],
            ArrayType.element_solid_strain: [n_states, n_solids, n_solid_layers, 6],
            ArrayType.element_solid_is_alive: [n_states, n_solids],
            ArrayType.element_solid_plastic_strain_tensor: [n_states, n_solids, n_solid_layers, 6],
            ArrayType.element_solid_thermal_strain_tensor: [n_states, n_solids, n_solid_layers, 6],
            # thick shells
            ArrayType.element_tshell_stress: [n_states, n_tshells, n_tshells_layers, 6],
            ArrayType.element_tshell_effective_plastic_strain: [
                n_states,
                n_tshells,
                n_tshells_layers,
            ],
            ArrayType.element_tshell_history_variables: [
                n_states,
                n_tshells,
                n_tshells_layers,
                n_tshells_history_vars,
            ],
            ArrayType.element_tshell_strain: [n_states, n_tshells, 2, 6],
            ArrayType.element_tshell_is_alive: [n_states, n_tshells],
            # beams
            ArrayType.element_beam_axial_force: [n_states, n_beams],
            ArrayType.element_beam_shear_force: [n_states, n_beams, 2],
            ArrayType.element_beam_bending_moment: [n_states, n_beams, 2],
            ArrayType.element_beam_torsion_moment: [n_states, n_beams],
            ArrayType.element_beam_shear_stress: [n_states, n_beams, n_beams_layers, 2],
            ArrayType.element_beam_axial_stress: [n_states, n_beams, n_beams_layers],
            ArrayType.element_beam_plastic_strain: [n_states, n_beams, n_beams_layers],
            ArrayType.element_beam_axial_strain: [n_states, n_beams, n_beams_layers],
            ArrayType.element_beam_history_vars: [
                n_states,
                n_beams,
                n_beams_layers + 3,
                n_beams_history_vars,
            ],
            ArrayType.element_beam_is_alive: [n_states, n_beams],
            # shells
            ArrayType.element_shell_stress: [n_states, n_shells_reduced, n_shell_layers, 6],
            ArrayType.element_shell_effective_plastic_strain: [
                n_states,
                n_shells_reduced,
                n_shell_layers,
            ],
            ArrayType.element_shell_history_vars: [
                n_states,
                n_shells_reduced,
                n_shell_layers,
                n_shell_history_vars,
            ],
            ArrayType.element_shell_bending_moment: [n_states, n_shells_reduced, 3],
            ArrayType.element_shell_shear_force: [n_states, n_shells_reduced, 2],
            ArrayType.element_shell_normal_force: [n_states, n_shells_reduced, 3],
            ArrayType.element_shell_thickness: [n_states, n_shells_reduced],
            ArrayType.element_shell_unknown_variables: [n_states, n_shells_reduced, 2],
            ArrayType.element_shell_internal_energy: [n_states, n_shells_reduced],
            ArrayType.element_shell_strain: [n_states, n_shells_reduced, 2, 6],
            ArrayType.element_shell_thermal_strain_tensor: [n_states, n_shells_reduced, 6],
            ArrayType.element_shell_plastic_strain_tensor: [
                n_states,
                n_shells_reduced,
                n_shell_layers,
                6,
            ],
            ArrayType.element_shell_is_alive: [n_states, n_shells],
            # sph
            ArrayType.sph_deletion: [n_states, n_sph_particles],
            ArrayType.sph_radius: [n_states, n_sph_particles],
            ArrayType.sph_pressure: [n_states, n_sph_particles],
            ArrayType.sph_stress: [n_states, n_sph_particles, 6],
            ArrayType.sph_effective_plastic_strain: [n_states, n_sph_particles],
            ArrayType.sph_density: [n_states, n_sph_particles],
            ArrayType.sph_internal_energy: [n_states, n_sph_particles],
            ArrayType.sph_n_neighbors: [n_states, n_sph_particles],
            ArrayType.sph_strain: [n_states, n_sph_particles, 6],
            ArrayType.sph_mass: [n_states, n_sph_particles],
            # airbag
            ArrayType.airbag_n_active_particles: [n_states, n_airbags],
            ArrayType.airbag_bag_volume: [n_states, n_airbags],
            ArrayType.airbag_particle_gas_id: [n_states, n_airbag_particles],
            ArrayType.airbag_particle_chamber_id: [n_states, n_airbag_particles],
            ArrayType.airbag_particle_leakage: [n_states, n_airbag_particles],
            ArrayType.airbag_particle_mass: [n_states, n_airbag_particles],
            ArrayType.airbag_particle_radius: [n_states, n_airbag_particles],
            ArrayType.airbag_particle_spin_energy: [n_states, n_airbag_particles],
            ArrayType.airbag_particle_translation_energy: [n_states, n_airbag_particles],
            ArrayType.airbag_particle_nearest_segment_distance: [n_states, n_airbag_particles],
            ArrayType.airbag_particle_position: [n_states, n_airbag_particles, 3],
            ArrayType.airbag_particle_velocity: [n_states, n_airbag_particles, 3],
            # rigid road
            ArrayType.rigid_road_displacement: [n_states, n_roads, 3],
            ArrayType.rigid_road_velocity: [n_states, n_roads, 3],
            # rigid body
            ArrayType.rigid_body_coordinates: [n_states, n_rigid_bodies, 3],
            ArrayType.rigid_body_rotation_matrix: [n_states, n_rigid_bodies, 9],
            ArrayType.rigid_body_velocity: [n_states, n_rigid_bodies, 3],
            ArrayType.rigid_body_rot_velocity: [n_states, n_rigid_bodies, 3],
            ArrayType.rigid_body_acceleration: [n_states, n_rigid_bodies, 3],
            ArrayType.rigid_body_rot_acceleration: [n_states, n_rigid_bodies, 3],
        }

        # only allocate available arrays
        if array_names is None:
            array_names = ArrayType.get_state_array_names()

        # BUGFIX
        # These arrays are actually integer types, all other state arrays
        # are floats
        int_state_arrays = [
            ArrayType.airbag_n_active_particles,
            ArrayType.airbag_particle_gas_id,
            ArrayType.airbag_particle_chamber_id,
            ArrayType.airbag_particle_leakage,
        ]

        # (2) ALLOCATE ARRAYS
        # this looper allocates the arrays specified by the user.
        for array_name in array_names:

            array_dtype = header.ftype if array_name not in int_state_arrays else header.itype

            if array_name in state_array_shapes:
                array_dict[array_name] = np.empty(state_array_shapes[array_name], dtype=array_dtype)
            else:
                raise ValueError(
                    f"Array '{array_name}' is not a state array. "
                    f"Please try one of: {list(state_array_shapes.keys())}",
                )

    @staticmethod
    def _read_states_transfer_memory(
        i_state: int, buffer_array_dict: dict, master_array_dict: dict
    ):
        """Transfers the memory from smaller buffer arrays with only a few
        timesteps into the major one

        Parameters
        ----------
        i_state: int
            current state index
        buffer_array_dict: dict
            dict with arrays of only a few timesteps
        master_array_dict: dict
            dict with the parent master arrays

        Notes
        -----
            If an array in the master dict is not found in the buffer dict
            then this array is set to `None`.
        """

        state_array_names = ArrayType.get_state_array_names()

        arrays_to_delete = []
        for array_name, array in master_array_dict.items():

            # copy memory to big array
            if array_name in buffer_array_dict:
                buffer_array = buffer_array_dict[array_name]
                n_states_buffer_array = buffer_array.shape[0]
                array[i_state : i_state + n_states_buffer_array] = buffer_array
            else:
                # remove unnecesary state arrays (not geometry arrays!)
                # we "could" deal with this in the allocate function
                # by not allocating them but this would replicate code
                # in the reading functions
                if array_name in state_array_names:
                    arrays_to_delete.append(array_name)

        for array_name in arrays_to_delete:
            del master_array_dict[array_name]

    def _compute_n_bytes_per_state(self) -> int:
        """Computes the number of bytes for every state

        Returns
        -------
        n_bytes_per_state: int
            number of bytes of every state
        """

        if not self.header:
            return 0

        # timestep
        timestep_offset = 1 * self.header.wordsize
        # global vars
        global_vars_offset = self.header.n_global_vars * self.header.wordsize
        # node vars
        n_node_vars = (
            self.header.has_node_displacement
            + self.header.has_node_velocity
            + self.header.has_node_acceleration
        ) * self.header.n_dimensions

        if self.header.has_node_temperatures:
            n_node_vars += 1
        if self.header.has_node_temperature_layers:
            n_node_vars += 2
        if self.header.has_node_heat_flux:
            n_node_vars += 3
        if self.header.has_node_mass_scaling:
            n_node_vars += 1
        if self.header.has_node_temperature_gradient:
            n_node_vars += 1
        if self.header.has_node_residual_forces:
            n_node_vars += 3
        if self.header.has_node_residual_moments:
            n_node_vars += 3

        node_data_offset = n_node_vars * self.header.n_nodes * self.header.wordsize
        # thermal shit
        therm_data_offset = (
            self.header.n_solid_thermal_vars * self.header.n_solids * self.header.wordsize
        )
        # solids
        solid_offset = self.header.n_solids * self.header.n_solid_vars * self.header.wordsize
        # tshells
        tshell_offset = (
            self.header.n_thick_shells * self.header.n_thick_shell_vars * self.header.wordsize
        )
        # beams
        beam_offset = self.header.n_beams * self.header.n_beam_vars * self.header.wordsize
        # shells
        shell_offset = (
            (self.header.n_shells - self._material_section_info.n_rigid_shells)
            * self.header.n_shell_vars
            * self.header.wordsize
        )
        # Manual
        # "NOTE: This CFDDATA is no longer output by ls-dyna."
        cfd_data_offset = 0
        # sph
        sph_offset = self.header.n_sph_nodes * self._sph_info.n_sph_vars * self.header.wordsize
        # deleted nodes and elems ... or nothing
        elem_deletion_offset = 0
        if self.header.has_node_deletion_data:
            elem_deletion_offset = self.header.n_nodes * self.header.wordsize
        elif self.header.has_element_deletion_data:
            elem_deletion_offset = (
                self.header.n_beams
                + self.header.n_shells
                + self.header.n_solids
                + self.header.n_thick_shells
            ) * self.header.wordsize
        # airbag particle offset
        if self._airbag_info.n_airbags:
            particle_state_offset = (
                self._airbag_info.n_airbags * self._airbag_info.n_airbag_state_variables
                + self._airbag_info.n_particles * self._airbag_info.n_particle_state_variables
            ) * self.header.wordsize
        else:
            particle_state_offset = 0
        # rigid road stuff whoever uses this
        road_surface_offset = self._rigid_road_info.n_roads * 6 * self.header.wordsize
        # rigid body motion data
        if self.header.has_rigid_body_data:
            n_rigids = self._rigid_body_info.n_rigid_bodies
            n_rigid_vars = 12 if self.header.has_reduced_rigid_body_data else 24
            rigid_body_motion_offset = n_rigids * n_rigid_vars * self.header.wordsize
        else:
            rigid_body_motion_offset = 0
        # ... not supported
        extra_data_offset = 0

        n_bytes_per_state = (
            timestep_offset
            + global_vars_offset
            + node_data_offset
            + therm_data_offset
            + solid_offset
            + tshell_offset
            + beam_offset
            + shell_offset
            + cfd_data_offset
            + sph_offset
            + elem_deletion_offset
            + particle_state_offset
            + road_surface_offset
            + rigid_body_motion_offset
            + extra_data_offset
        )
        return n_bytes_per_state

    def _read_states(self, filepath: str):
        """Read the states from the d3plot

        Parameters
        ----------
        filepath: str
            path to the d3plot
        """

        if not self._buffer or not filepath:
            self._state_info.n_timesteps = 0
            return

        # (0) OFFSETS
        bytes_per_state = self._compute_n_bytes_per_state()

        # load the memory from the files
        self.bb_generator = self._read_d3plot_file_generator(
            self.buffered_reading, self.state_filter
        )

        # (1) READ STATE DATA
        n_states = next(self.bb_generator)

        # determine whether to transfer arrays
        if not self.buffered_reading:
            transfer_arrays = False
        else:
            transfer_arrays = True
        if self.state_filter is not None and any(self.state_filter):
            transfer_arrays = True
        if self.state_array_filter:
            transfer_arrays = True

        # arrays need to be preallocated if we transfer them
        if transfer_arrays:
            self._read_states_allocate_arrays(
                self.header,
                self._material_section_info,
                self._airbag_info,
                self._rigid_road_info,
                self._rigid_body_info,
                n_states,
                self._n_rigid_walls,
                self._n_parts,
                self.state_array_filter,
                self.arrays,
            )

        i_state = 0
        for bb_states, n_states in self.bb_generator:

            # dictionary to store the temporary, partial arrays
            # if we do not transfer any arrays we store them directly
            # in the classes main dict
            array_dict = {} if transfer_arrays else self.arrays

            # sometimes there is just a geometry in the file
            if n_states == 0:
                continue

            # state data as array
            array_length = int(n_states) * int(bytes_per_state)
            state_data = bb_states.read_ndarray(0, array_length, 1, self.header.ftype)
            state_data = state_data.reshape((n_states, -1))

            var_index = 0

            # global state header
            var_index = self._read_states_global_section(state_data, var_index, array_dict)

            # node data
            var_index = self._read_states_nodes(state_data, var_index, array_dict)

            # thermal solid data
            var_index = self._read_states_solids_thermal(state_data, var_index, array_dict)

            # cfddata was originally here

            # solids
            var_index = self._read_states_solids(state_data, var_index, array_dict)

            # tshells
            var_index = self._read_states_tshell(state_data, var_index, array_dict)

            # beams
            var_index = self._read_states_beams(state_data, var_index, array_dict)

            # shells
            var_index = self._read_states_shell(state_data, var_index, array_dict)

            # element and node deletion info
            var_index = self._read_states_is_alive(state_data, var_index, array_dict)

            # sph
            var_index = self._read_states_sph(state_data, var_index, array_dict)

            # airbag particle data
            var_index = self._read_states_airbags(state_data, var_index, array_dict)

            # road surface data
            var_index = self._read_states_road_surfaces(state_data, var_index, array_dict)

            # rigid body motion
            var_index = self._read_states_rigid_body_motion(state_data, var_index, array_dict)

            # transfer memory
            if transfer_arrays:
                self._read_states_transfer_memory(i_state, array_dict, self.arrays)

            # increment state counter
            i_state += n_states
            self._state_info.n_timesteps = i_state

        if transfer_arrays:
            self._buffer = None
            self.bb_states = None

    def _read_states_global_section(
        self, state_data: np.ndarray, var_index: int, array_dict: dict
    ) -> int:
        """Read the global vars for the state

        Parameters
        ----------
        state_data: np.ndarray
            array with entire state data
        var_index: int
            variable index in the state data array
        array_dict: dict
            dictionary to store the loaded arrays in

        Returns
        -------
        var_index: int
            updated variable index after reading the section
        """

        # we wrap globals, parts and rigid walls into a single try
        # catch block since in the header the global section is
        # defined by those three. If we fail in any of those we can
        # only heal by skipping all together and jumping forward
        original_var_index = var_index
        try:
            # timestep
            array_dict[ArrayType.global_timesteps] = state_data[:, var_index]
            var_index += 1

            # global stuff
            var_index = self._read_states_globals(state_data, var_index, array_dict)

            # parts
            var_index = self._read_states_parts(state_data, var_index, array_dict)

            # rigid walls
            var_index = self._read_states_rigid_walls(state_data, var_index, array_dict)

        except Exception:
            # print
            trb_msg = traceback.format_exc()
            msg = "A failure in %s was caught:\n%s"
        finally:
            timestep_var_size = 1
            var_index = original_var_index + self.header.n_global_vars + timestep_var_size

        return var_index

    def _read_states_globals(self, state_data: np.ndarray, var_index: int, array_dict: dict) -> int:
        """Read the part data in the state section

        Parameters
        ----------
        state_data: np.ndarray
            array with entire state data
        var_index: int
            variable index in the state data array
        array_dict: dict
            dictionary to store the loaded arrays in

        Returns
        -------
        var_index: int
            updated variable index after reading the section
        """

        n_global_vars = self.header.n_global_vars

        # global stuff
        i_global_var = 0
        if i_global_var < n_global_vars:
            array_dict[ArrayType.global_kinetic_energy] = state_data[:, var_index + i_global_var]
            i_global_var += 1
        if i_global_var < n_global_vars:
            array_dict[ArrayType.global_internal_energy] = state_data[:, var_index + i_global_var]
            i_global_var += 1
        if i_global_var < n_global_vars:
            array_dict[ArrayType.global_total_energy] = state_data[:, var_index + i_global_var]
            i_global_var += 1
        if i_global_var + 3 <= n_global_vars:
            array_dict[ArrayType.global_velocity] = state_data[
                :, var_index + i_global_var : var_index + i_global_var + 3
            ]
            i_global_var += 3

        return var_index + i_global_var

    def _read_states_parts(self, state_data: np.ndarray, var_index: int, array_dict: dict) -> int:
        """Read the part data in the state section

        Parameters
        ----------
        state_data: np.ndarray
            array with entire state data
        var_index: int
            variable index in the state data array
        array_dict: dict
            dictionary to store the loaded arrays in

        Returns
        -------
        var_index: int
            updated variable index after reading the section
        """

        n_states = state_data.shape[0]
        timestep_word = 1
        n_global_vars = self.header.n_global_vars + timestep_word

        # part infos
        # n_parts = self._n_parts
        n_parts = self.header.n_parts

        # part internal energy
        if var_index + n_parts <= n_global_vars:
            array_dict[ArrayType.part_internal_energy] = state_data[
                :, var_index : var_index + n_parts
            ]
            var_index += n_parts

        # part kinetic energy
        if var_index + n_parts <= n_global_vars:
            array_dict[ArrayType.part_kinetic_energy] = state_data[
                :, var_index : var_index + n_parts
            ]
            var_index += n_parts

        # part velocity
        if var_index + 3 * n_parts <= n_global_vars:
            array_dict[ArrayType.part_velocity] = state_data[
                :, var_index : var_index + 3 * n_parts
            ].reshape((n_states, n_parts, 3))
            var_index += 3 * n_parts

        # part mass
        if var_index + n_parts <= n_global_vars:
            array_dict[ArrayType.part_mass] = state_data[:, var_index : var_index + n_parts]
            var_index += n_parts

        # part hourglass energy
        if var_index + n_parts <= n_global_vars:
            array_dict[ArrayType.part_hourglass_energy] = state_data[
                :, var_index : var_index + n_parts
            ]
            var_index += n_parts

        return var_index

    def _read_states_rigid_walls(
        self, state_data: np.ndarray, var_index: int, array_dict: dict
    ) -> int:
        """Read the rigid wall data in the state section

        Parameters
        ----------
        state_data: np.ndarray
            array with entire state data
        var_index: int
            variable index in the state data array
        array_dict: dict
            dictionary to store the loaded arrays in

        Returns
        -------
        var_index: int
            updated variable index after reading the section
        """

        n_states = state_data.shape[0]

        i_global_var = 6 + 7 * self.header.n_parts
        n_global_vars = self.header.n_global_vars

        # rigid walls
        previous_global_vars = i_global_var
        n_rigid_wall_vars = 4 if self.header.version >= 971 else 1
        # +1 is timestep which is not considered a global var ... seriously
        n_rigid_walls = self._n_rigid_walls
        if n_global_vars >= previous_global_vars + n_rigid_walls * n_rigid_wall_vars:
            if (
                previous_global_vars + n_rigid_walls * n_rigid_wall_vars
                != self.header.n_global_vars
            ):
                var_index += self.header.n_global_vars - previous_global_vars
            else:

                # rigid wall force
                if n_rigid_walls * n_rigid_wall_vars != 0:
                    array_dict[ArrayType.rigid_wall_force] = state_data[
                        :, var_index : var_index + n_rigid_walls
                    ]
                    var_index += n_rigid_walls

                    # rigid wall position
                    if n_rigid_wall_vars > 1:
                        array_dict[ArrayType.rigid_wall_position] = state_data[
                            :, var_index : var_index + 3 * n_rigid_walls
                        ].reshape(n_states, n_rigid_walls, 3)
                        var_index += 3 * n_rigid_walls

        return var_index

    def _read_states_nodes(self, state_data: np.ndarray, var_index: int, array_dict: dict) -> int:
        """Read the node data in the state section

        Parameters
        ----------
        state_data: np.ndarray
            array with entire state data
        var_index: int
            variable index in the state data array
        array_dict: dict
            dictionary to store the loaded arrays in

        Returns
        -------
        var_index: int
            updated variable index after reading the section
        """

        if self.header.n_nodes <= 0:
            return var_index

        n_dim = self.header.n_dimensions
        n_states = state_data.shape[0]
        n_nodes = self.header.n_nodes

        # displacement
        if self.header.has_node_displacement:
            try:
                tmp_array = state_data[:, var_index : var_index + n_dim * n_nodes].reshape(
                    (n_states, n_nodes, n_dim)
                )
                array_dict[ArrayType.node_displacement] = tmp_array
            except Exception:
                trb_msg = traceback.format_exc()
                msg = "A failure in %s was caught:\n%s"
            finally:
                var_index += n_dim * n_nodes

        # temperatures
        if self.header.has_node_temperatures:

            # only node temperatures
            if not self.header.has_node_temperature_layers:
                try:
                    array_dict[ArrayType.node_temperature] = state_data[
                        :, var_index : var_index + n_nodes
                    ]
                except Exception:
                    trb_msg = traceback.format_exc()
                    msg = "A failure in %s was caught:\n%s"
                finally:
                    var_index += n_nodes
            # node temperature layers
            else:
                try:
                    tmp_array = state_data[:, var_index : var_index + 3 * n_nodes].reshape(
                        (n_states, n_nodes, 3)
                    )
                    array_dict[ArrayType.node_temperature] = tmp_array
                except Exception:
                    trb_msg = traceback.format_exc()
                    msg = "A failure in %s was caught:\n%s"
                finally:
                    var_index += 3 * n_nodes

        # node heat flux
        if self.header.has_node_heat_flux:
            try:
                tmp_array = state_data[:, var_index : var_index + 3 * n_nodes].reshape(
                    (n_states, n_nodes, 3)
                )
                array_dict[ArrayType.node_heat_flux] = tmp_array
            except Exception:
                trb_msg = traceback.format_exc()
                msg = "A failure in %s was caught:\n%s"
            finally:
                var_index += 3 * n_nodes

        # mass scaling
        if self.header.has_node_mass_scaling:
            try:
                array_dict[ArrayType.node_mass_scaling] = state_data[
                    :, var_index : var_index + n_nodes
                ]
            except Exception:
                trb_msg = traceback.format_exc()
                msg = "A failure in %s was caught:\n%s"
            finally:
                var_index += n_nodes

        # node temperature gradient
        # Unclear: verify (could also be between temperature and node heat flux)
        if self.header.has_node_temperature_gradient:
            try:
                array_dict[ArrayType.node_temperature_gradient] = state_data[
                    :, var_index : var_index + n_nodes
                ]
            except Exception:
                trb_msg = traceback.format_exc()
                msg = "A failure in %s was caught:\n%s"
            finally:
                var_index += n_nodes

        # node residual forces and moments
        # Unclear: verify (see before, according to docs this is after previous)
        if self.header.has_node_residual_forces:
            try:
                array_dict[ArrayType.node_residual_forces] = state_data[
                    :, var_index : var_index + 3 * n_nodes
                ].reshape((n_states, n_nodes, 3))
            except Exception:
                trb_msg = traceback.format_exc()
                msg = "A failure in %s was caught:\n%s"
            finally:
                var_index += n_nodes * 3

        if self.header.has_node_residual_moments:
            try:
                array_dict[ArrayType.node_residual_moments] = state_data[
                    :, var_index : var_index + 3 * n_nodes
                ].reshape((n_states, n_nodes, 3))
            except Exception:
                trb_msg = traceback.format_exc()
                msg = "A failure in %s was caught:\n%s"
            finally:
                var_index += n_nodes * 3

        # velocity
        if self.header.has_node_velocity:
            try:
                tmp_array = state_data[:, var_index : var_index + n_dim * n_nodes].reshape(
                    (n_states, n_nodes, n_dim)
                )
                array_dict[ArrayType.node_velocity] = tmp_array
            except Exception:
                trb_msg = traceback.format_exc()
                msg = "A failure in %s was caught:\n%s"
            finally:
                var_index += n_dim * n_nodes

        # acceleration
        if self.header.has_node_acceleration:
            try:
                tmp_array = state_data[:, var_index : var_index + n_dim * n_nodes].reshape(
                    (n_states, n_nodes, n_dim)
                )
                array_dict[ArrayType.node_acceleration] = tmp_array
            except Exception:
                trb_msg = traceback.format_exc()
                msg = "A failure in %s was caught:\n%s"
            finally:
                var_index += n_dim * n_nodes

        return var_index

    def _read_states_solids_thermal(
        self, state_data: np.ndarray, var_index: int, array_dict: dict
    ) -> int:
        """Read the thermal data for solids

        Parameters
        ----------
        state_data: np.ndarray
            array with entire state data
        var_index: int
            variable index in the state data array
        array_dict: dict
            dictionary to store the loaded arrays in

        Returns
        -------
        var_index: int
            updated variable index after reading the section
        """

        if self.header.n_solid_thermal_vars <= 0:
            return var_index

        n_states = state_data.shape[0]
        n_solids = self.header.n_solids
        n_thermal_vars = self.header.n_solid_thermal_vars

        try:
            tmp_array = state_data[:, var_index : var_index + n_solids * n_thermal_vars]
            array_dict[ArrayType.element_solid_thermal_data] = tmp_array.reshape(
                (n_states, n_solids, n_thermal_vars)
            )
        except Exception:
            trb_msg = traceback.format_exc()
            msg = "A failure in %s was caught:\n%s"
        finally:
            var_index += n_thermal_vars * n_solids

        return var_index

    def _read_states_solids(self, state_data: np.ndarray, var_index: int, array_dict: dict) -> int:
        """Read the state data of the solid elements

        Parameters
        ----------
        state_data: np.ndarray
            array with entire state data
        var_index: int
            variable index in the state data array
        array_dict: dict
            dictionary to store the loaded arrays in

        Returns
        -------
        var_index: int
            updated variable index after reading the section
        """

        if self.header.n_solids <= 0 or self.header.n_solid_vars <= 0:
            return var_index

        n_solid_vars = self.header.n_solid_vars
        n_solids = self.header.n_solids
        n_states = state_data.shape[0]
        n_strain_vars = 6 * self.header.has_element_strain
        n_history_vars = self.header.n_solid_history_vars
        n_solid_layers = self.header.n_solid_layers

        # double safety here, if either the formatting of the solid state data
        # or individual arrays fails then we catch it
        try:
            # this is a sanity check if the manual was understood correctly
            #
            # NOTE due to plotcompress we disable this check, it can delete
            # variables so that stress or pstrain might be missing despite
            # being always present in the file spec
            #
            # n_solid_vars2 = (7 +
            #                  n_history_vars)

            # if n_solid_vars2 != n_solid_vars:
            #     msg = "n_solid_vars != n_solid_vars_computed: {} != {}."\
            #           + " Solid variables might be wrong."

            solid_state_data = state_data[
                :, var_index : var_index + n_solid_vars * n_solids
            ].reshape((n_states, n_solids, n_solid_layers, n_solid_vars // n_solid_layers))

            i_solid_var = 0

            # stress
            try:
                if self.header.has_solid_stress:
                    array_dict[ArrayType.element_solid_stress] = solid_state_data[:, :, :, :6]
            except Exception:
                trb_msg = traceback.format_exc()
                msg = "A failure in %s was caught:\n%s"
            finally:
                i_solid_var += 6 * self.header.has_solid_stress

            # effective plastic strain
            try:
                # in case plotcompress deleted stresses but pstrain exists
                if self.header.has_solid_pstrain:
                    array_dict[ArrayType.element_solid_effective_plastic_strain] = solid_state_data[
                        :, :, :, i_solid_var
                    ].reshape((n_states, n_solids, n_solid_layers))
            except Exception:
                trb_msg = traceback.format_exc()
                msg = "A failure in %s was caught:\n%s"
            finally:
                i_solid_var += 1 * self.header.has_solid_pstrain

            # history vars
            if n_history_vars:
                try:
                    array_dict[ArrayType.element_solid_history_variables] = solid_state_data[
                        :, :, :, i_solid_var : i_solid_var + n_history_vars
                    ]
                except Exception:
                    trb_msg = traceback.format_exc()
                    msg = "A failure in %s was caught:\n%s"
                finally:
                    i_solid_var += n_history_vars

            # strain
            # they are the last 6 entries of the history vars
            if n_strain_vars:
                try:
                    array_dict[ArrayType.element_solid_strain] = array_dict[
                        ArrayType.element_solid_history_variables
                    ][:, :, :, -n_strain_vars:]

                    array_dict[ArrayType.element_solid_history_variables] = array_dict[
                        ArrayType.element_solid_history_variables
                    ][:, :, :, :-n_strain_vars]

                    if not all(array_dict[ArrayType.element_solid_history_variables].shape):
                        del array_dict[ArrayType.element_solid_history_variables]
                except Exception:
                    trb_msg = traceback.format_exc()
                    msg = "A failure in %s was caught:\n%s"

            # plastic strain tensor
            if self.header.has_solid_shell_plastic_strain_tensor:
                try:
                    array_dict[ArrayType.element_solid_plastic_strain_tensor] = solid_state_data[
                        :, :, :, i_solid_var : i_solid_var + 6
                    ]
                except Exception:
                    trb_msg = traceback.format_exc()
                    msg = "A failure in %s was caught:\n%s"
                finally:
                    i_solid_var += 6

            # thermal strain tensor
            if self.header.has_solid_shell_thermal_strain_tensor:
                try:
                    array_dict[ArrayType.element_solid_thermal_strain_tensor] = solid_state_data[
                        :, :, i_solid_var : i_solid_var + 6
                    ]
                except Exception:
                    trb_msg = traceback.format_exc()
                    msg = "A failure in %s was caught:\n%s"
                finally:
                    i_solid_var += 6

        # catch formatting in solid_state_datra
        except Exception:
            trb_msg = traceback.format_exc()
            msg = "A failure in %s was caught:\n%s"
        # always increment variable count
        finally:
            var_index += n_solids * n_solid_vars

        return var_index

    def _read_states_tshell(self, state_data: np.ndarray, var_index: int, array_dict: dict) -> int:
        """Read the state data for thick shell elements

        Parameters
        ----------
        state_data: np.ndarray
            array with entire state data
        var_index: int
            variable index in the state data array
        array_dict: dict
            dictionary to store the loaded arrays in

        Returns
        -------
        var_index: int
            updated variable index after reading the section
        """

        if self.header.n_thick_shells <= 0 or self.header.n_thick_shell_vars <= 0:
            return var_index

        n_states = state_data.shape[0]
        n_tshells = self.header.n_thick_shells
        n_history_vars = self.header.n_shell_tshell_history_vars
        n_layers = self.header.n_shell_tshell_layers
        n_layer_vars = n_layers * (
            6 * self.header.has_shell_tshell_stress
            + self.header.has_shell_tshell_pstrain
            + n_history_vars
        )
        n_strain_vars = 12 * self.header.has_element_strain
        n_thsell_vars = self.header.n_thick_shell_vars
        has_stress = self.header.has_shell_tshell_stress
        has_pstrain = self.header.has_shell_tshell_pstrain

        try:
            # this is a sanity check if the manual was understood correctly
            n_tshell_vars2 = n_layer_vars + n_strain_vars

            if n_tshell_vars2 != n_thsell_vars:
                msg = (
                    "n_tshell_vars != n_tshell_vars_computed: %d != %d."
                    " Thick shell variables might be wrong."
                )

            # thick shell element data
            tshell_data = state_data[:, var_index : var_index + n_thsell_vars * n_tshells]
            tshell_data = tshell_data.reshape((n_states, n_tshells, n_thsell_vars))

            # extract layer data
            tshell_layer_data = tshell_data[:, :, slice(0, n_layer_vars)]
            tshell_layer_data = tshell_layer_data.reshape((n_states, n_tshells, n_layers, -1))
            tshell_nonlayer_data = tshell_data[:, :, n_layer_vars:]

            # STRESS
            i_tshell_layer_var = 0
            if has_stress:
                try:
                    array_dict[ArrayType.element_tshell_stress] = tshell_layer_data[
                        :, :, :, i_tshell_layer_var : i_tshell_layer_var + 6
                    ].reshape((n_states, n_tshells, n_layers, 6))
                except Exception:
                    trb_msg = traceback.format_exc()
                    msg = "A failure in %d was caught:\n%s"
                finally:
                    i_tshell_layer_var += 6

            # PSTRAIN
            if has_pstrain:
                try:
                    array_dict[
                        ArrayType.element_tshell_effective_plastic_strain
                    ] = tshell_layer_data[:, :, :, i_tshell_layer_var].reshape(
                        (n_states, n_tshells, n_layers)
                    )
                except Exception:
                    trb_msg = traceback.format_exc()
                    msg = "A failure in %s was caught:\n%s"
                finally:
                    i_tshell_layer_var += 1

            # HISTORY VARS
            if n_history_vars:
                try:
                    array_dict[ArrayType.element_tshell_history_variables] = tshell_layer_data[
                        :, :, :, i_tshell_layer_var : i_tshell_layer_var + n_history_vars
                    ].reshape((n_states, n_tshells, n_layers, n_history_vars))
                except Exception:
                    trb_msg = traceback.format_exc()
                    msg = "A failure in %s was caught:\n%s"

            # STRAIN (only non layer data for tshells)
            if n_strain_vars:
                try:
                    tshell_nonlayer_data = tshell_nonlayer_data[:, :, :n_strain_vars]
                    array_dict[ArrayType.element_tshell_strain] = tshell_nonlayer_data.reshape(
                        (n_states, n_tshells, 2, 6)
                    )
                except Exception:
                    trb_msg = traceback.format_exc()
                    msg = "A failure in %s was caught:\n%s"

        except Exception:
            trb_msg = traceback.format_exc()
            msg = "A failure in %s was caught:\n%s"
        finally:
            var_index += n_thsell_vars * n_tshells

        return var_index

    def _read_states_beams(self, state_data: np.ndarray, var_index: int, array_dict: dict) -> int:
        """Read the state data for beams

        Parameters
        ----------
        state_data: np.ndarray
            array with entire state data
        var_index: int
            variable index in the state data array
        array_dict: dict
            dictionary to store the loaded arrays in

        Returns
        -------
        var_index: int
            updated variable index after reading the section
        """

        if self.header.n_beams <= 0 or self.header.n_beam_vars <= 0:
            return var_index

        # usual beam vars
        # pylint: disable = invalid-name
        N_BEAM_BASIC_VARS = 6
        # beam intergration point vars
        # pylint: disable = invalid-name
        N_BEAM_IP_VARS = 5

        n_states = state_data.shape[0]
        n_beams = self.header.n_beams
        n_history_vars = self.header.n_beam_history_vars
        n_beam_vars = self.header.n_beam_vars
        n_layers = int(
            (-3 * n_history_vars + n_beam_vars - N_BEAM_BASIC_VARS)
            / (n_history_vars + N_BEAM_IP_VARS)
        )
        # n_layer_vars = 6 + N_BEAM_IP_VARS * n_layers
        n_layer_vars = N_BEAM_IP_VARS * n_layers

        try:
            # beam element data
            beam_data = state_data[:, var_index : var_index + n_beam_vars * n_beams]
            beam_data = beam_data.reshape((n_states, n_beams, n_beam_vars))

            # extract layer data
            beam_nonlayer_data = beam_data[:, :, :N_BEAM_BASIC_VARS]
            beam_layer_data = beam_data[:, :, N_BEAM_BASIC_VARS : N_BEAM_BASIC_VARS + n_layer_vars]
            beam_layer_data = beam_layer_data.reshape((n_states, n_beams, n_layers, N_BEAM_IP_VARS))

            # axial force
            try:
                array_dict[ArrayType.element_beam_axial_force] = beam_nonlayer_data[
                    :, :, 0
                ].reshape((n_states, n_beams))
            except Exception:
                trb_msg = traceback.format_exc()
                msg = "A failure in %s was caught:\n%s"

            # shear force
            try:
                array_dict[ArrayType.element_beam_shear_force] = beam_nonlayer_data[
                    :, :, 1:3
                ].reshape((n_states, n_beams, 2))
            except Exception:
                trb_msg = traceback.format_exc()
                msg = "A failure in %s was caught:\n%s"

            # bending moment
            try:
                array_dict[ArrayType.element_beam_bending_moment] = beam_nonlayer_data[
                    :, :, 3:5
                ].reshape((n_states, n_beams, 2))
            except Exception:
                trb_msg = traceback.format_exc()
                msg = "A failure in %s was caught:\n%s"

            # torsion moment
            try:
                array_dict[ArrayType.element_beam_torsion_moment] = beam_nonlayer_data[
                    :, :, 5
                ].reshape((n_states, n_beams))
            except Exception:
                trb_msg = traceback.format_exc()
                msg = "A failure in %s was caught:\n%s"

            if n_layers:

                # BUGFIX?
                # According to the database manual the first
                # two layer vars are the shear stress and then
                # axial stress. Tests with FEMZIP and META though
                # suggests that axial stress comes first.

                # axial stress
                try:
                    array_dict[ArrayType.element_beam_axial_stress] = beam_layer_data[:, :, :, 0]
                except Exception:
                    trb_msg = traceback.format_exc()
                    msg = "A failure in %s was caught:\n%s"

                # shear stress
                try:
                    array_dict[ArrayType.element_beam_shear_stress] = beam_layer_data[:, :, :, 1:3]
                except Exception:
                    trb_msg = traceback.format_exc()
                    msg = "A failure in %s was caught:\n%s"

                # eff. plastic strain
                try:
                    array_dict[ArrayType.element_beam_plastic_strain] = beam_layer_data[:, :, :, 3]
                except Exception:
                    trb_msg = traceback.format_exc()
                    msg = "A failure in %s was caught:\n%s"

                # axial strain
                try:
                    array_dict[ArrayType.element_beam_axial_strain] = beam_layer_data[:, :, :, 4]
                except Exception:
                    trb_msg = traceback.format_exc()
                    msg = "A failure in %s was caught:\n%s"

            # history vars
            if n_history_vars:
                try:
                    array_dict[ArrayType.element_beam_history_vars] = beam_data[
                        :, :, 6 + n_layer_vars :
                    ].reshape((n_states, n_beams, 3 + n_layers, n_history_vars))
                except Exception:
                    trb_msg = traceback.format_exc()
                    msg = "A failure in %s was caught:\n%s"

        # failure of formatting beam state data
        except Exception:
            trb_msg = traceback.format_exc()
            msg = "A failure in %s was caught:\n%s"
        # always increment variable index
        finally:
            var_index += n_beams * n_beam_vars


        return var_index

    def _read_states_shell(self, state_data: np.ndarray, var_index: int, array_dict: dict) -> int:
        """Read the state data for shell elements

        Parameters
        ----------
        state_data: np.ndarray
            array with entire state data
        var_index: int
            variable index in the state data array
        array_dict: dict
            dictionary to store the loaded arrays in

        Returns
        -------
        var_index: int
            updated variable index after reading the section
        """

        # bugfix
        #
        # Interestingly, dyna seems to write result values for rigid shells in
        # the d3part file, but not in the d3plot. Of course this is not
        # documented ...
        n_reduced_shells = (
            self.header.n_shells
            if self.header.filetype == D3plotFiletype.D3PART
            else self.header.n_shells - self._material_section_info.n_rigid_shells
        )

        if self.header.n_shell_vars <= 0 or n_reduced_shells <= 0:
            return var_index

        n_states = state_data.shape[0]
        n_shells = n_reduced_shells
        n_shell_vars = self.header.n_shell_vars

        # what is in the file?
        n_layers = self.header.n_shell_tshell_layers
        n_history_vars = self.header.n_shell_tshell_history_vars
        n_stress_vars = 6 * self.header.has_shell_tshell_stress
        n_pstrain_vars = 1 * self.header.has_shell_tshell_pstrain
        n_force_variables = 8 * self.header.has_shell_forces
        n_extra_variables = 4 * self.header.has_shell_extra_variables
        n_strain_vars = 12 * self.header.has_element_strain
        n_plastic_strain_tensor = 6 * n_layers * self.header.has_solid_shell_plastic_strain_tensor
        n_thermal_strain_tensor = 6 * self.header.has_solid_shell_thermal_strain_tensor

        try:
            # this is a sanity check if the manual was understood correctly
            n_shell_vars2 = (
                n_layers * (n_stress_vars + n_pstrain_vars + n_history_vars)
                + n_force_variables
                + n_extra_variables
                + n_strain_vars
                + n_plastic_strain_tensor
                + n_thermal_strain_tensor
            )

            if n_shell_vars != n_shell_vars2:
                msg = (
                    "n_shell_vars != n_shell_vars_computed: %d != %d."
                    " Shell variables might be wrong."
                )

            n_layer_vars = n_layers * (n_stress_vars + n_pstrain_vars + n_history_vars)

            # shell element data
            shell_data = state_data[:, var_index : var_index + n_shell_vars * n_shells]
            shell_data = shell_data.reshape((n_states, n_shells, n_shell_vars))

            # extract layer data
            shell_layer_data = shell_data[:, :, :n_layer_vars]
            shell_layer_data = shell_layer_data.reshape((n_states, n_shells, n_layers, -1))
            shell_nonlayer_data = shell_data[:, :, n_layer_vars:]

            # save layer stuff
            # STRESS
            layer_var_index = 0
            if n_stress_vars:
                try:
                    array_dict[ArrayType.element_shell_stress] = shell_layer_data[
                        :, :, :, :n_stress_vars
                    ].reshape((n_states, n_shells, n_layers, n_stress_vars))
                except Exception:
                    trb_msg = traceback.format_exc()
                    msg = "A failure in %s was caught:\n%s"
                finally:
                    layer_var_index += n_stress_vars

            # PSTRAIN
            if n_pstrain_vars:
                try:
                    array_dict[ArrayType.element_shell_effective_plastic_strain] = shell_layer_data[
                        :, :, :, layer_var_index
                    ].reshape((n_states, n_shells, n_layers))
                except Exception:
                    trb_msg = traceback.format_exc()
                    msg = "A failure in %s was caught:\n%s"
                finally:
                    layer_var_index += 1

            # HISTORY VARIABLES
            if n_history_vars:
                try:
                    array_dict[ArrayType.element_shell_history_vars] = shell_layer_data[
                        :, :, :, layer_var_index : layer_var_index + n_history_vars
                    ].reshape((n_states, n_shells, n_layers, n_history_vars))
                except Exception:
                    trb_msg = traceback.format_exc()
                    msg = "A failure in %s was caught:\n%s"
                finally:
                    layer_var_index += n_history_vars

            # save nonlayer stuff
            # forces
            nonlayer_var_index = 0
            if n_force_variables:
                try:
                    array_dict[ArrayType.element_shell_bending_moment] = shell_nonlayer_data[
                        :, :, 0:3
                    ].reshape((n_states, n_shells, 3))
                    array_dict[ArrayType.element_shell_shear_force] = shell_nonlayer_data[
                        :, :, 3:5
                    ].reshape((n_states, n_shells, 2))
                    array_dict[ArrayType.element_shell_normal_force] = shell_nonlayer_data[
                        :, :, 5:8
                    ].reshape((n_states, n_shells, 3))
                except Exception:
                    trb_msg = traceback.format_exc()
                    msg = "A failure in %s was caught:\n%s"
                finally:
                    nonlayer_var_index += n_force_variables

            # weird stuff
            if n_extra_variables:
                try:
                    array_dict[ArrayType.element_shell_thickness] = shell_nonlayer_data[
                        :, :, nonlayer_var_index
                    ].reshape((n_states, n_shells))
                    array_dict[ArrayType.element_shell_unknown_variables] = shell_nonlayer_data[
                        :, :, nonlayer_var_index + 1 : nonlayer_var_index + 3
                    ].reshape((n_states, n_shells, 2))
                except Exception:
                    trb_msg = traceback.format_exc()
                    msg = "A failure in %s was caught:\n%s"
                finally:
                    nonlayer_var_index += 3

            # strain present
            if n_strain_vars:
                try:
                    shell_strain = shell_nonlayer_data[
                        :, :, nonlayer_var_index : nonlayer_var_index + n_strain_vars
                    ]
                    array_dict[ArrayType.element_shell_strain] = shell_strain.reshape(
                        (n_states, n_shells, 2, 6)
                    )
                except Exception:
                    trb_msg = traceback.format_exc()
                    msg = "A failure in %s was caught:\n%s"
                finally:
                    nonlayer_var_index += n_strain_vars

            # internal energy is behind strain if strain is written
            if self.header.has_shell_extra_variables:
                try:
                    array_dict[ArrayType.element_shell_internal_energy] = shell_nonlayer_data[
                        :, :, nonlayer_var_index
                    ].reshape((n_states, n_shells))
                except Exception:
                    trb_msg = traceback.format_exc()
                    msg = "A failure in %s was caught:\n%s"

            # PLASTIC STRAIN TENSOR
            if n_plastic_strain_tensor:
                try:
                    pstrain_tensor = shell_nonlayer_data[
                        :, :, nonlayer_var_index : nonlayer_var_index + n_plastic_strain_tensor
                    ]
                    array_dict[
                        ArrayType.element_shell_plastic_strain_tensor
                    ] = pstrain_tensor.reshape((n_states, n_shells, n_layers, 6))
                except Exception:
                    trb_msg = traceback.format_exc()
                    msg = "A failure in %s was caught:\n%s"
                finally:
                    nonlayer_var_index += n_plastic_strain_tensor

            # THERMAL STRAIN TENSOR
            if n_thermal_strain_tensor:
                try:
                    thermal_tensor = shell_nonlayer_data[
                        :, :, nonlayer_var_index : nonlayer_var_index + n_thermal_strain_tensor
                    ]
                    array_dict[
                        ArrayType.element_shell_thermal_strain_tensor
                    ] = thermal_tensor.reshape((n_states, n_shells, 6))
                except Exception:
                    trb_msg = traceback.format_exc()
                    msg = "A failure in %s was caught:\n%s"
                finally:
                    nonlayer_var_index += n_thermal_strain_tensor

        # error in formatting shell state data
        except Exception:
            trb_msg = traceback.format_exc()
            msg = "A failure in %s was caught:\n%s"

        # always increment variable index
        finally:
            var_index += n_shell_vars * n_shells

        return var_index

    def _read_states_is_alive(
        self, state_data: np.ndarray, var_index: int, array_dict: dict
    ) -> int:
        """Read deletion info for nodes, elements, etc

        Parameters
        ----------
        state_data: np.ndarray
            array with entire state data
        var_index: int
            variable index in the state data array
        array_dict: dict
            dictionary to store the loaded arrays in

        Returns
        -------
        var_index: int
            updated variable index after reading the section
        """

        if not self.header.has_node_deletion_data and not self.header.has_element_deletion_data:
            return var_index

        n_states = state_data.shape[0]

        # NODES
        if self.header.has_node_deletion_data:
            n_nodes = self.header.n_nodes

            if n_nodes > 0:
                try:
                    array_dict[ArrayType.node_is_alive] = state_data[
                        :, var_index : var_index + n_nodes
                    ]
                except Exception:
                    trb_msg = traceback.format_exc()
                    msg = "A failure in %s was caught:\n%s"
                finally:
                    var_index += n_nodes

        # element deletion info
        elif self.header.has_element_deletion_data:
            n_solids = self.header.n_solids
            n_tshells = self.header.n_thick_shells
            n_shells = self.header.n_shells
            n_beams = self.header.n_beams
            # n_elems = n_solids + n_tshells + n_shells + n_beams

            # SOLIDS
            if n_solids > 0:
                try:
                    array_dict[ArrayType.element_solid_is_alive] = state_data[
                        :, var_index : var_index + n_solids
                    ].reshape((n_states, n_solids))
                except Exception:
                    trb_msg = traceback.format_exc()
                    msg = "A failure in %s was caught:\n%s"
                finally:
                    var_index += n_solids

            # TSHELLS
            if n_tshells > 0:
                try:
                    array_dict[ArrayType.element_tshell_is_alive] = state_data[
                        :, var_index : var_index + n_tshells
                    ].reshape((n_states, n_tshells))
                except Exception:
                    trb_msg = traceback.format_exc()
                    msg = "A failure in %s was caught:\n%s"
                finally:
                    var_index += n_tshells

            # SHELLS
            if n_shells > 0:
                try:
                    array_dict[ArrayType.element_shell_is_alive] = state_data[
                        :, var_index : var_index + n_shells
                    ].reshape((n_states, n_shells))
                except Exception:
                    trb_msg = traceback.format_exc()
                    msg = "A failure in %s was caught:\n%s"
                finally:
                    var_index += n_shells

            # BEAMS
            if n_beams > 0:
                try:
                    array_dict[ArrayType.element_beam_is_alive] = state_data[
                        :, var_index : var_index + n_beams
                    ].reshape((n_states, n_beams))
                except Exception:
                    trb_msg = traceback.format_exc()
                    msg = "A failure in %s was caught:\n%s"
                finally:
                    var_index += n_beams

        return var_index

    def _read_states_sph(self, state_data: np.ndarray, var_index: int, array_dict: dict) -> int:
        """Read the sph state data

        Parameters
        ----------
        state_data: np.ndarray
            array with entire state data
        var_index: int
            variable index in the state data array
        array_dict: dict
            dictionary to store the loaded arrays in

        Returns
        -------
        var_index: int
            updated variable index after reading the section
        """

        if self.header.n_sph_nodes <= 0:
            return var_index

        info = self._sph_info
        n_states = state_data.shape[0]
        n_particles = self.header.n_sph_nodes
        n_variables = info.n_sph_vars

        # extract data
        try:
            sph_data = state_data[:, var_index : var_index + n_particles * n_variables]

            i_var = 1

            # deletion
            try:
                array_dict[ArrayType.sph_deletion] = sph_data[:, 0] < 0
            except Exception:
                trb_msg = traceback.format_exc()
                msg = "A failure in %s was caught:\n%s"

            # particle radius
            if info.has_influence_radius:
                try:
                    array_dict[ArrayType.sph_radius] = sph_data[:, i_var]
                except Exception:
                    trb_msg = traceback.format_exc()
                    msg = "A failure in %s was caught:\n%s"
                finally:
                    i_var += 1

            # pressure
            if info.has_particle_pressure:
                try:
                    array_dict[ArrayType.sph_pressure] = sph_data[:, i_var]
                except Exception:
                    trb_msg = traceback.format_exc()
                    msg = "A failure in %s was caught:\n%s"
                finally:
                    i_var += 1

            # stress
            if info.has_stresses:
                try:
                    array_dict[ArrayType.sph_stress] = sph_data[
                        :, i_var : i_var + n_particles * 6
                    ].reshape((n_states, n_particles, 6))
                except Exception:
                    trb_msg = traceback.format_exc()
                    msg = "A failure in %s was caught:\n%s"
                finally:
                    i_var += 6 * n_particles

            # eff. plastic strain
            if info.has_plastic_strain:
                try:
                    array_dict[ArrayType.sph_effective_plastic_strain] = sph_data[:, i_var]
                except Exception:
                    trb_msg = traceback.format_exc()
                    msg = "A failure in %s was caught:\n%s"
                finally:
                    i_var += 1

            # density
            if info.has_material_density:
                try:
                    array_dict[ArrayType.sph_density] = sph_data[:, i_var]
                except Exception:
                    trb_msg = traceback.format_exc()
                    msg = "A failure in %s was caught:\n%s"
                finally:
                    i_var += 1

            # internal energy
            if info.has_internal_energy:
                try:
                    array_dict[ArrayType.sph_internal_energy] = sph_data[:, i_var]
                except Exception:
                    trb_msg = traceback.format_exc()
                    msg = "A failure in %s was caught:\n%s"
                finally:
                    i_var += 1

            # number of neighbors
            if info.has_n_affecting_neighbors:
                try:
                    array_dict[ArrayType.sph_n_neighbors] = sph_data[:, i_var]
                except Exception:
                    trb_msg = traceback.format_exc()
                    msg = "A failure in %s was caught:\n%s"
                finally:
                    i_var += 1

            # strain and strainrate
            if info.has_strain_and_strainrate:

                try:
                    array_dict[ArrayType.sph_strain] = sph_data[
                        :, i_var : i_var + n_particles * 6
                    ].reshape((n_states, n_particles, 6))
                except Exception:
                    trb_msg = traceback.format_exc()
                    msg = "A failure in %s was caught:\n%s"
                finally:
                    i_var += 6 * n_particles

                try:
                    array_dict[ArrayType.sph_strainrate] = sph_data[
                        :, i_var : i_var + n_particles * 6
                    ].reshape((n_states, n_particles, 6))
                except Exception:
                    trb_msg = traceback.format_exc()
                    msg = "A failure in %s was caught:\n%s"
                finally:
                    i_var += 6 * n_particles

            # mass
            if info.has_mass:
                try:
                    array_dict[ArrayType.sph_mass] = sph_data[:, i_var]
                except Exception:
                    trb_msg = traceback.format_exc()
                    msg = "A failure in %s was caught:\n%s"
                finally:
                    i_var += 1

        except Exception:
            trb_msg = traceback.format_exc()
            msg = "A failure in %s was caught:\n%s"
        finally:
            var_index += n_particles * n_variables

        return var_index

    def _read_states_airbags(self, state_data: np.ndarray, var_index: int, array_dict: dict) -> int:
        """Read the airbag state data

        Parameters
        ----------
        state_data: np.ndarray
            array with entire state data
        var_index: int
            variable index in the state data array
        array_dict: dict
            dictionary to store the loaded arrays in

        Returns
        -------
        var_index: int
            updated variable index after reading the section
        """

        if self.header.n_airbags <= 0:
            return var_index

        n_states = state_data.shape[0]
        info = self._airbag_info
        n_airbag_geom_vars = info.n_geometric_variables
        n_airbags = info.n_airbags
        n_state_airbag_vars = info.n_airbag_state_variables
        n_particles = info.n_particles
        n_particle_vars = info.n_particle_state_variables

        # Warning
        # I am not sure if this is right ...
        n_total_vars = n_airbags * n_state_airbag_vars + n_particles * n_particle_vars

        try:
            # types
            # nlist = ngeom + nvar + nstgeom
            airbag_var_types = self.arrays[ArrayType.airbag_variable_types]
            airbag_var_names = self.arrays[ArrayType.airbag_variable_names]
            # geom_var_types = airbag_var_types[:n_airbag_geom_vars]
            particle_var_types = airbag_var_types[
                n_airbag_geom_vars : n_airbag_geom_vars + n_particle_vars
            ]
            particle_var_names = airbag_var_names[
                n_airbag_geom_vars : n_airbag_geom_vars + n_particle_vars
            ]

            airbag_state_var_types = airbag_var_types[n_airbag_geom_vars + n_particle_vars :]
            airbag_state_var_names = airbag_var_names[n_airbag_geom_vars + n_particle_vars :]

            # required for dynamic reading
            def get_dtype(type_flag):
                return self._header.itype if type_flag == 1 else self.header.ftype

            # extract airbag data
            airbag_state_data = state_data[:, var_index : var_index + n_total_vars]

            # airbag data
            airbag_data = airbag_state_data[:, : n_airbags * n_state_airbag_vars].reshape(
                (n_states, n_airbags, n_state_airbag_vars)
            )
            airbag_state_offset = n_airbags * n_state_airbag_vars

            # particle data
            particle_data = airbag_state_data[
                :, airbag_state_offset : airbag_state_offset + n_particles * n_particle_vars
            ].reshape((n_states, n_particles, n_particle_vars))

            # save sh...

            # airbag state vars
            for i_airbag_state_var in range(n_state_airbag_vars):
                var_name = airbag_state_var_names[i_airbag_state_var].strip()
                var_type = airbag_state_var_types[i_airbag_state_var]

                if var_name.startswith("Act Gas"):
                    try:
                        array_dict[ArrayType.airbag_n_active_particles] = airbag_data[
                            :, :, i_airbag_state_var
                        ].view(get_dtype(var_type))
                    except Exception:
                        trb_msg = traceback.format_exc()
                        msg = "A failure in %s was caught:\n%s"
                elif var_name.startswith("Bag Vol"):
                    try:
                        array_dict[ArrayType.airbag_bag_volume] = airbag_data[
                            :, :, i_airbag_state_var
                        ].view(get_dtype(var_type))
                    except Exception:
                        trb_msg = traceback.format_exc()
                        msg = "A failure in %s was caught:\n%s"
                else:
                    warn_msg = "Unknown airbag state var: '%s'. Skipping it."

            # particles yay
            for i_particle_var in range(n_particle_vars):
                var_type = particle_var_types[i_particle_var]
                var_name = particle_var_names[i_particle_var].strip()

                # particle gas id
                if var_name.startswith("GasC ID"):
                    try:
                        array_dict[ArrayType.airbag_particle_gas_id] = particle_data[
                            :, :, i_particle_var
                        ].view(get_dtype(var_type))
                    except Exception:
                        trb_msg = traceback.format_exc()
                        msg = "A failure in %s %s was caught:\n%s"
                # particle chamber id
                elif var_name.startswith("Cham ID"):
                    try:
                        array_dict[ArrayType.airbag_particle_chamber_id] = particle_data[
                            :, :, i_particle_var
                        ].view(get_dtype(var_type))
                    except Exception:
                        trb_msg = traceback.format_exc()
                        msg = "A failure in %s %s was caught:\n%s"
                # particle leakage
                elif var_name.startswith("Leakage"):
                    try:
                        array_dict[ArrayType.airbag_particle_leakage] = particle_data[
                            :, :, i_particle_var
                        ].view(get_dtype(var_type))
                    except Exception:
                        trb_msg = traceback.format_exc()
                        msg = "A failure in %s %s was caught:\n%s"
                # particle mass
                elif var_name.startswith("Mass"):
                    try:
                        array_dict[ArrayType.airbag_particle_mass] = particle_data[
                            :, :, i_particle_var
                        ].view(get_dtype(var_type))
                    except Exception:
                        trb_msg = traceback.format_exc()
                        msg = "A failure in %s %s was caught:\n%s"
                    # particle radius
                    try:
                        array_dict[ArrayType.airbag_particle_radius] = particle_data[
                            :, :, i_particle_var
                        ].view(get_dtype(var_type))
                    except Exception:
                        trb_msg = traceback.format_exc()
                        msg = "A failure in %s %s was caught:\n%s"
                # particle spin energy
                elif var_name.startswith("Spin En"):
                    try:
                        array_dict[ArrayType.airbag_particle_spin_energy] = particle_data[
                            :, :, i_particle_var
                        ].view(get_dtype(var_type))
                    except Exception:
                        trb_msg = traceback.format_exc()
                        msg = "A failure in %s %s was caught:\n%s"
                # particle translational energy
                elif var_name.startswith("Tran En"):
                    try:
                        array_dict[ArrayType.airbag_particle_translation_energy] = particle_data[
                            :, :, i_particle_var
                        ].view(get_dtype(var_type))
                    except Exception:
                        trb_msg = traceback.format_exc()
                        msg = "A failure in %s %s was caught:\n%s"
                # particle segment distance
                elif var_name.startswith("NS dist"):
                    try:
                        array_dict[
                            ArrayType.airbag_particle_nearest_segment_distance
                        ] = particle_data[:, :, i_particle_var].view(get_dtype(var_type))
                    except Exception:
                        trb_msg = traceback.format_exc()
                        msg = "A failure in %s %s was caught:\n%s"
                # particle position
                elif var_name.startswith("Pos x"):
                    try:
                        particle_var_names_stripped = [
                            entry.strip() for entry in particle_var_names
                        ]
                        i_particle_var_x = i_particle_var
                        i_particle_var_y = particle_var_names_stripped.index("Pos y")
                        i_particle_var_z = particle_var_names_stripped.index("Pos z")

                        array_dict[ArrayType.airbag_particle_position] = particle_data[
                            :, :, (i_particle_var_x, i_particle_var_y, i_particle_var_z)
                        ].view(get_dtype(var_type))
                    except Exception:
                        trb_msg = traceback.format_exc()
                        msg = "A failure in %s %s was caught:\n%s"
                elif var_name.startswith("Pos y"):
                    # handled in Pos x
                    pass
                elif var_name.startswith("Pos z"):
                    # handled in Pos x
                    pass
                # particle velocity
                elif var_name.startswith("Vel x"):
                    try:
                        particle_var_names_stripped = [
                            entry.strip() for entry in particle_var_names
                        ]
                        i_particle_var_x = i_particle_var
                        i_particle_var_y = particle_var_names_stripped.index("Vel y")
                        i_particle_var_z = particle_var_names_stripped.index("Vel z")

                        array_dict[ArrayType.airbag_particle_velocity] = particle_data[
                            :, :, (i_particle_var_x, i_particle_var_y, i_particle_var_z)
                        ].view(get_dtype(var_type))
                    except Exception:
                        trb_msg = traceback.format_exc()
                        msg = "A failure in %s %s was caught:\n%s"

        except Exception:
            trb_msg = traceback.format_exc()
            msg = "A failure in %s was caught:\n%s"
        finally:
            var_index += n_total_vars

        return var_index

    def _read_states_road_surfaces(
        self, state_data: np.ndarray, var_index: int, array_dict: dict
    ) -> int:
        """Read the road surfaces state data for whoever wants this ...

        Parameters
        ----------
        state_data: np.ndarray
            array with entire state data
        var_index: int
            variable index in the state data array
        array_dict: dict
            dictionary to store the loaded arrays in

        Returns
        -------
        var_index: int
            updated variable index after reading the section
        """

        if not self.header.has_rigid_road_surface:
            return var_index

        n_states = state_data.shape[0]
        info = self._rigid_road_info
        n_roads = info.n_roads

        try:
            # read road data
            road_data = state_data[:, var_index : var_index + 6 * n_roads].reshape(
                (n_states, n_roads, 2, 3)
            )

            # DISPLACEMENT
            try:
                array_dict[ArrayType.rigid_road_displacement] = road_data[:, :, 0, :]
            except Exception:
                trb_msg = traceback.format_exc()
                msg = "A failure in %s was caught:\n%s"

            # VELOCITY
            try:
                array_dict[ArrayType.rigid_road_velocity] = road_data[:, :, 1, :]
            except Exception:
                trb_msg = traceback.format_exc()
                msg = "A failure in %s was caught:\n%s"

        except Exception:
            trb_msg = traceback.format_exc()
            msg = "A failure in %s was caught:\n%s"
        finally:
            var_index += 6 * n_roads

        return var_index

    def _read_states_rigid_body_motion(
        self, state_data: np.ndarray, var_index: int, array_dict: dict
    ) -> int:
        """Read the road surfaces state data for whoever want this ...

        Parameters
        ----------
        state_data: np.ndarray
            array with entire state data
        var_index: int
            variable index in the state data array
        array_dict: dict
            dictionary to store the loaded arrays in

        Returns
        -------
        var_index: int
            updated variable index after reading the section
        """

        if not self.header.has_rigid_body_data:
            return var_index

        info = self._rigid_body_info
        n_states = state_data.shape[0]
        n_rigids = info.n_rigid_bodies
        n_rigid_vars = 12 if self.header.has_reduced_rigid_body_data else 24

        try:
            # do the thing
            rigid_body_data = state_data[
                :, var_index : var_index + n_rigids * n_rigid_vars
            ].reshape((n_states, n_rigids, n_rigid_vars))

            # let the party begin
            # rigid coordinates
            try:
                array_dict[ArrayType.rigid_body_coordinates] = rigid_body_data[:, :, :3]
            except Exception:
                trb_msg = traceback.format_exc()
                msg = "A failure in %s was caught:\n%s"
            finally:
                i_var = 3

            # rotation matrix
            try:
                array_dict[ArrayType.rigid_body_rotation_matrix] = rigid_body_data[
                    :, :, i_var : i_var + 9
                ]
            except Exception:
                trb_msg = traceback.format_exc()
                msg = "A failure in %s was caught:\n%s"
            finally:
                i_var += 9

            if self.header.has_reduced_rigid_body_data:
                return var_index

            # velocity pewpew
            try:
                array_dict[ArrayType.rigid_body_velocity] = rigid_body_data[:, :, i_var : i_var + 3]
            except Exception:
                trb_msg = traceback.format_exc()
                msg = "A failure in %s was caught:\n%s"
            finally:
                i_var += 3

            # rotational velocity
            try:
                array_dict[ArrayType.rigid_body_rot_velocity] = rigid_body_data[
                    :, :, i_var : i_var + 3
                ]
            except Exception:
                trb_msg = traceback.format_exc()
                msg = "A failure in %s was caught:\n%s"
            finally:
                i_var += 3

            # acceleration
            try:
                array_dict[ArrayType.rigid_body_acceleration] = rigid_body_data[
                    :, :, i_var : i_var + 3
                ]
            except Exception:
                trb_msg = traceback.format_exc()
                msg = "A failure in %s was caught:\n%s"
            finally:
                i_var += 3

            # rotational acceleration
            try:
                array_dict[ArrayType.rigid_body_rot_acceleration] = rigid_body_data[
                    :, :, i_var : i_var + 3
                ]
            except Exception:
                trb_msg = traceback.format_exc()
                msg = "A failure in %s was caught:\n%s"
            finally:
                i_var += 3

        except Exception:
            trb_msg = traceback.format_exc()
            msg = "A failure in %s was caught:\n%s"

        finally:
            var_index += n_rigids * n_rigid_vars

        return var_index

    def _collect_file_infos(self, size_per_state: int) -> List[MemoryInfo]:
        """This routine collects the memory and file info for the d3plot files

        Parameters
        ----------
        size_per_state: int
            size of every state to be read

        Returns
        -------
        memory_infos: List[MemoryInfo]
            memory infos about the states

        Notes
        -----
            State data is expected directly behind geometry data
            Unfortunately data is spread across multiple files.
            One file could contain geometry and state data but states
            may also be littered accross several files. This would
            not be an issue, if dyna would not always write in blocks
            of 512 words of memory, leaving zero byte padding blocks
            at the end of files. These need to be removed and/or taken
            care of.
        """

        if not self._buffer:
            return []

        base_filepath = self.header.filepath

        # bugfix
        # If you encounter these int casts more often here this is why:
        # Some ints around here are numpy.int32 which can overflow
        # (sometimes there is a warning ... sometimes not ...)
        # we cast to python ints in order to prevent overflow.
        size_per_state = int(size_per_state)

        # Info:
        #
        # We need to determine here how many states are in every file
        # without really loading the file itself. For big files this is
        # simply filesize // state_size.
        # For files though with a smaller filesize this may cause issues
        # e.g.
        # filesize 2048 bytes (minimum filesize from dyna)
        # geom_size 200 bytes
        # state_size 200 bytes
        # File contains:
        # -> 1 state * state_size + geom_size = 400 bytes
        # Wrong State Estimation:
        # -> (filesize - geom_size) // state_size = 9 states != 1 state
        #
        # To avoid this wrong number of states when reading small files
        # we need to search the end mark (here nonzero byte) from the rear
        # of the file.
        # This though needs the file to be loaded into memory. To make this
        # very light, we simply memorymap a small fraction of the file starting
        # from the rear until we have our nonzero byte. Since the end mark
        # is usually in the first block loaded, there should not be any performance
        # concerns, even with bigger files.

        # query for state files
        filepaths = D3plot._find_dyna_result_files(base_filepath)

        # compute state data in first file
        # search therefore the first non-zero byte from the rear
        last_nonzero_byte_index = self._buffer.size
        mview_inv_arr = np.asarray(self._buffer.memoryview[::-1])
        # pylint: disable = invalid-name
        BLOCK_SIZE = 2048
        for start in range(0, self._buffer.size, BLOCK_SIZE):
            (nz_indexes,) = np.nonzero(mview_inv_arr[start : start + BLOCK_SIZE])
            if len(nz_indexes):
                last_nonzero_byte_index = self._buffer.size - (start + nz_indexes[0])
                break
        n_states_beyond_geom = (
            last_nonzero_byte_index - self.geometry_section_size
        ) // size_per_state

        # bugfix: if states are too big we can get a negative estimation
        n_states_beyond_geom = max(0, n_states_beyond_geom)

        # memory required later
        memory_infos = [
            MemoryInfo(
                start=self.geometry_section_size,  # type: ignore
                length=n_states_beyond_geom * size_per_state,  # type: ignore
                filepath=base_filepath,
                n_states=n_states_beyond_geom,  # type: ignore
                filesize=self._buffer.size,
                use_mmap=True,
            )
        ]

        # compute amount of state data in every further file
        for filepath in filepaths:
            filesize = os.path.getsize(filepath)
            last_nonzero_byte_index = -1

            n_blocks = filesize // mmap.ALLOCATIONGRANULARITY
            rest_size = filesize % mmap.ALLOCATIONGRANULARITY
            block_length = mmap.ALLOCATIONGRANULARITY
            with open(filepath, "rb") as fp:

                # search last rest block (page-aligned)
                # page-aligned means the start must be
                # a multiple of mmap.ALLOCATIONGRANULARITY
                # otherwise we get an error on linux
                if rest_size:
                    start = n_blocks * block_length
                    mview = memoryview(
                        mmap.mmap(
                            fp.fileno(), offset=start, length=rest_size, access=mmap.ACCESS_READ
                        ).read()
                    )
                    (nz_indexes,) = np.nonzero(mview[::-1])
                    if len(nz_indexes):
                        last_nonzero_byte_index = start + rest_size - nz_indexes[0]

                # search in blocks from the reair
                if last_nonzero_byte_index == -1:
                    for i_block in range(n_blocks - 1, -1, -1):
                        start = block_length * i_block
                        mview = memoryview(
                            mmap.mmap(
                                fp.fileno(),
                                offset=start,
                                length=block_length,
                                access=mmap.ACCESS_READ,
                            ).read()
                        )
                        (nz_indexes,) = np.nonzero(mview[::-1])
                        if len(nz_indexes):
                            index = block_length - nz_indexes[0]
                            last_nonzero_byte_index = start + index
                            break

            if last_nonzero_byte_index == -1:
                msg = "The file {0} seems to be missing it's endmark."
                raise RuntimeError(msg.format(filepath))

            # BUGFIX
            # In d3eigv it could be observed that there is not necessarily an end mark.
            # As a consequence the last byte can indeed be zero. We control this by
            # checking if the last nonzero byte was smaller than the state size which
            # makes no sense.
            if (
                self.header.filetype == D3plotFiletype.D3EIGV
                and last_nonzero_byte_index < size_per_state <= filesize
            ):
                last_nonzero_byte_index = size_per_state

            n_states_in_file = last_nonzero_byte_index // size_per_state
            memory_infos.append(
                MemoryInfo(
                    start=0,
                    length=size_per_state * (n_states_in_file),
                    filepath=filepath,
                    n_states=n_states_in_file,
                    filesize=filesize,
                    use_mmap=False,
                )
            )

        return memory_infos

    @staticmethod
    def _read_file_from_memory_info(
        memory_infos: Union[MemoryInfo, List[MemoryInfo]]
    ) -> Tuple[BinaryBuffer, int]:
        """Read files from a single or multiple memory infos

        Parameters
        ----------
        memory_infos: MemoryInfo or List[MemoryInfo]
            memory infos for loading a file (see `D3plot._collect_file_infos`)

        Returns
        -------
        bb_states: BinaryBuffer
            New binary buffer with all states perfectly linear in memory
        n_states: int
            Number of states to be expected

        Notes
        -----
            This routine in contrast to `D3plot._read_state_bytebuffer` is used
            to load only a fraction of files into memory.
        """

        # single file case
        if isinstance(memory_infos, MemoryInfo):
            memory_infos = [memory_infos]

        # allocate memory
        # bugfix: casting to int prevents int32 overflow for large files
        memory_required = 0
        for mem in memory_infos:
            memory_required += int(mem.length)
        mview = memoryview(bytearray(memory_required))

        # transfer memory for other files
        n_states = 0
        total_offset = 0
        for minfo in memory_infos:
            with open(minfo.filepath, "br") as fp:
                # NOTE
                # mmap is too slow but maybe there are faster
                # ways to use mmap correctly
                # if minfo.use_mmap:

                #     # memory mapping can only be done page aligned
                #     mmap_start = (minfo.start // mmap.ALLOCATIONGRANULARITY) * \
                #         mmap.ALLOCATIONGRANULARITY
                #     mview_start = minfo.start - mmap_start

                #     end = minfo.start + minfo.length
                #     n_end_pages = (end // mmap.ALLOCATIONGRANULARITY +
                #                    (end % mmap.ALLOCATIONGRANULARITY != 0))
                #     mmap_length = n_end_pages * mmap.ALLOCATIONGRANULARITY - mmap_start
                #     if mmap_start + mmap_length > minfo.filesize:
                #         mmap_length = minfo.filesize - mmap_start

                #     with mmap.mmap(fp.fileno(),
                #                    length=mmap_length,
                #                    offset=mmap_start,
                #                    access=mmap.ACCESS_READ) as mp:
                #         # mp.seek(mview_start)
                #         # mview[total_offset:total_offset +
                #         #       minfo.length] = mp.read(minfo.length)

                #         mview[total_offset:total_offset +
                #               minfo.length] = mp[mview_start:mview_start + minfo.length]

                # else:
                fp.seek(minfo.start)
                fp.readinto(mview[total_offset : total_offset + minfo.length])  # type: ignore

            total_offset += minfo.length
            n_states += minfo.n_states

        # save
        bb_states = BinaryBuffer()
        bb_states.memoryview = mview

        return bb_states, n_states

    def _read_state_bytebuffer(self, size_per_state: int):
        """This routine reads the data for state information

        Parameters
        ----------
        size_per_state: int
            size of every state to be read

        Returns
        -------
        bb_states: BinaryBuffer
            New binary buffer with all states perfectly linear in memory
        n_states: int
            Number of states to be expected

        Notes
        -----
            State data is expected directly behind geometry data
            Unfortunately data is spread across multiple files.
            One file could contain geometry and state data but states
            may also be littered accross several files. This would
            not be an issue, if dyna would not always write in blocks
            of 512 words of memory, leaving zero byte padding blocks
            at the end of files. These need to be removed and/or taken
            care of.
        """

        if not self._buffer:
            return BinaryBuffer(), 0

        memory_infos = self._collect_file_infos(size_per_state)

        # allocate memory
        # bugfix: casting to int prevents int32 overflow for large files
        memory_required = 0
        for mem in memory_infos:
            memory_required += int(mem.length)
        mview = memoryview(bytearray(memory_required))

        # transfer memory from first file
        n_states = memory_infos[0].n_states
        start = memory_infos[0].start
        length = memory_infos[0].length
        end = start + length
        mview[:length] = self._buffer.memoryview[start:end]

        # transfer memory for other files
        total_offset = length
        for minfo in memory_infos[1:]:
            with open(minfo.filepath, "br") as fp:
                fp.seek(minfo.start)
                fp.readinto(mview[total_offset : total_offset + length])  # type: ignore

            total_offset += length
            n_states += minfo.n_states

        # save
        bb_states = BinaryBuffer()
        bb_states.memoryview = mview
        return bb_states, n_states

    @staticmethod
    def _find_dyna_result_files(filepath: str):
        """Searches all dyna result files

        Parameters
        ----------
        filepath: str
            path to the first basic d3plot file

        Returns
        -------
        filepaths: list of str
            path to all dyna files

        Notes
        -----
            The dyna files usually follow a scheme to
            simply have the base name and numbers appended
            e.g. (d3plot, d3plot0001, d3plot0002, etc.)
        """

        file_dir = os.path.dirname(filepath)
        file_dir = file_dir if len(file_dir) != 0 else "."
        file_basename = os.path.basename(filepath)

        pattern = f"({file_basename})[0-9]+$"
        reg = re.compile(pattern)

        filepaths = [
            os.path.join(file_dir, path)
            for path in os.listdir(file_dir)
            if os.path.isfile(os.path.join(file_dir, path)) and reg.match(path)
        ]

        # alphasort files to handle d3plots with more than 100 files
        # e.g. d3plot01, d3plot02, ..., d3plot100
        def convert(text):
            return int(text) if text.isdigit() else text.lower()

        number_pattern = "([0-9]+)"

        def alphanum_key(key):
            return [convert(c) for c in re.split(number_pattern, key)]

        return sorted(filepaths, key=alphanum_key)

    def _determine_wordsize(self):
        """Determine the precision of the file

        Returns
        -------
        wordsize: int
            size of each word in bytes
        """

        if not self._buffer:
            return 4, np.int32, np.float32

        # test file type flag (1=d3plot, 5=d3part, 11=d3eigv)

        # single precision
        value = self._buffer.read_number(44, np.int32)
        if value > 1000:
            value -= 1000
        if value in (1, 5, 11):
            return 4, np.int32, np.float32

        # double precision
        value = self._buffer.read_number(88, np.int64)
        if value > 1000:
            value -= 1000
        if value in (1, 5, 11):
            return 8, np.int64, np.float64

        raise RuntimeError(f"Unknown file type '{value}'.")

    def check_array_dims(
        self, array_dimensions: Dict[str, int], dimension_name: str, dimension_size: int = -1
    ):
        """This function checks if multiple arrays share an array dimensions
        with the same size.

        Parameters
        ----------
        array_dimensions: Dict[str, int]
            Array name and expected number of dimensions as dict
        dimension_name: str
            Name of the array dimension for error messages
        dimension_size: int
            Optional expected size. If not set then all entries must equal
            the first value collected.

        Raises
        ------
        ValueError
            If dimensions do not match in any kind of way.
        """

        dimension_size_dict = {}

        # collect all dimensions
        for typename, dimension_index in array_dimensions.items():
            if typename not in self.arrays:
                continue

            array = self.arrays[typename]

            if dimension_index >= array.ndim:
                msg = (
                    f"Array '{typename}' requires at least "
                    f"{dimension_index} dimensions ({dimension_name})"
                )
                raise ValueError(msg)

            dimension_size_dict[typename] = array.shape[dimension_index]

        # static dimension
        if dimension_size >= 0:
            arrays_with_wrong_dims = {
                typename: size
                for typename, size in dimension_size_dict.items()
                if size != dimension_size
            }

            if arrays_with_wrong_dims:
                msg = "The dimension %s of the following arrays is expected to have size %d:\n%s"
                msg_arrays = [
                    f" - name: {typename} dim: {array_dimensions[typename]} size: {size}"
                    for typename, size in arrays_with_wrong_dims.items()
                ]
                raise ValueError(msg, dimension_name, dimension_size, "\n".join(msg_arrays))

        # dynamic dimensions
        else:
            if dimension_size_dict:
                unique_sizes = np.unique(list(dimension_size_dict.values()))
                if len(unique_sizes) > 1:
                    msg = "Inconsistency in array dim '%d' detected:\n%s"
                    size_list = [
                        f"   - name: {typename}, dim: {array_dimensions[typename]}, size: {size}"
                        for typename, size in dimension_size_dict.items()
                    ]
                    raise ValueError(msg, dimension_name, "\n".join(size_list))
                if len(unique_sizes) == 1:
                    dimension_size = unique_sizes[0]

        if dimension_size < 0:
            return 0

        return dimension_size

    @staticmethod
    def _compare_n_bytes_checksum(n_bytes_written: int, n_bytes_expected: int):
        """Throw if the byte checksum was not ok

        Parameters
        ----------
        n_bytes_written: int
            bytes written to the file
        n_bytes_expected: int
            bytes expected from the header computation

        Raises
        ------
        RuntimeError
            If the byte count doesn't match.
        """
        if n_bytes_expected != n_bytes_written:
            msg = (
                "byte checksum wrong: "
                f"{n_bytes_expected} (header) != {n_bytes_written} (checksum)"
            )
            raise RuntimeError(msg)

    def _get_zero_byte_padding(self, n_bytes_written: int, block_size_bytes: int):
        """Compute the zero byte-padding at the end of files

        Parameters
        ----------
        n_bytes_written: int
            number of bytes already written to file
        block_size_bytes: int
            byte block size of the file

        Returns
        -------
        zero_bytes: bytes
            zero-byte padding ready to be written to the file
        """

        if block_size_bytes > 0:
            remaining_bytes = n_bytes_written % block_size_bytes
            n_bytes_to_fill = block_size_bytes - remaining_bytes if remaining_bytes != 0 else 0
            return b"\x00" * n_bytes_to_fill

        return b""

    def compare(self, d3plot2, array_eps: Union[float, None] = None):
        """Compare two d3plots and print the info

        Parameters
        ----------
        d3plot2: D3plot
            second d3plot
        array_eps: float or None
            tolerance for arrays

        Returns
        -------
        hdr_differences: dict
            differences in the header
        array_differences: dict
            difference between arrays as message

        Examples
        --------
            Comparison of a femzipped file and an uncompressed file. Femzip
            is a lossy compression, thus precision is traded for memory.

            >>> d3plot1 = D3plot("path/to/d3plot")
            >>> d3plot2 = D3plot("path/to/d3plot.fz")
            >>> hdr_diff, array_diff = d3plot1.compare(d3plot2)
            >>> for arraytype, msg in array_diff.items():
            >>>     print(name, msg)
            node_coordinates Δmax = 0.050048828125
            node_displacement Δmax = 0.050048828125
            node_velocity Δmax = 0.050048828125
            node_acceleration Δmax = 49998984.0
            element_beam_axial_force Δmax = 6.103515625e-05
            element_shell_stress Δmax = 0.0005035400390625
            element_shell_thickness Δmax = 9.999999717180685e-10
            element_shell_unknown_variables Δmax = 0.0005000010132789612
            element_shell_internal_energy Δmax = 188.41957092285156

        """

        # pylint: disable = too-many-nested-blocks

        assert isinstance(d3plot2, D3plot)
        d3plot1 = self

        hdr_differences = d3plot1.header.compare(d3plot2.header)

        # ARRAY COMPARISON
        array_differences = {}

        array_names = list(d3plot1.arrays.keys()) + list(d3plot2.arrays.keys())

        for name in array_names:

            array1 = (
                d3plot1.arrays[name] if name in d3plot1.arrays else "Array is missing in original"
            )

            array2 = d3plot2.arrays[name] if name in d3plot2.arrays else "Array is missing in other"

            # d3parts write results for rigid shells.
            # when rewriting as d3plot we simply
            # don't write the part_material_types
            # array which is the same as having no
            # rigid shells.
            d3plot1_is_d3part = d3plot1.header.filetype == D3plotFiletype.D3PART
            d3plot2_is_d3part = d3plot2.header.filetype == D3plotFiletype.D3PART
            if name == "part_material_type" and (d3plot1_is_d3part or d3plot2_is_d3part):
                continue

            # we have an array to compare
            if isinstance(array1, str):
                array_differences[name] = array1
            elif isinstance(array2, str):
                array_differences[name] = array2
            elif isinstance(array2, np.ndarray):
                comparison = False

                # compare arrays
                if isinstance(array1, np.ndarray):
                    if array1.shape != array2.shape:
                        comparison = f"shape mismatch {array1.shape} != {array2.shape}"
                    else:
                        if np.issubdtype(array1.dtype, np.number) and np.issubdtype(
                            array2.dtype, np.number
                        ):
                            diff = np.abs(array1 - array2)
                            if diff.size:
                                if array_eps is not None:
                                    diff2 = diff[diff > array_eps]
                                    if diff2.size:
                                        diff2_max = diff2.max()
                                        if diff2_max:
                                            comparison = f"Δmax = {diff2_max}"
                                else:
                                    diff_max = diff.max()
                                    if diff_max:
                                        comparison = f"Δmax = {diff_max}"
                        else:
                            n_mismatches = (array1 != array2).sum()
                            if n_mismatches:
                                comparison = f"Mismatches: {n_mismatches}"

                else:
                    comparison = "Arrays don't match"

                # print
                if comparison:
                    array_differences[name] = comparison

        return hdr_differences, array_differences

    def get_part_filter(
        self, filter_type: FilterType, part_ids: Iterable[int], for_state_array: bool = True
    ) -> np.ndarray:
        """Get a part filter for different entities

        Parameters
        ----------
        filter_type: lasso.dyna.FilterType
            the array type to filter for (beam, shell, solid, tshell, node)
        part_ids: Iterable[int]
            part ids to filter out
        for_state_array: bool
            if the filter is meant for a state array. Makes a difference
            for shells if rigid bodies are in the model (mattyp == 20)

        Returns
        -------
        mask: np.ndarray
            mask usable on arrays to filter results

        Examples
        --------
            >>> from lasso.dyna import D3plot, ArrayType, FilterType
            >>> d3plot = D3plot("path/to/d3plot")
            >>> part_ids = [13, 14]
            >>> mask = d3plot.get_part_filter(FilterType.shell)
            >>> shell_stress = d3plot.arrays[ArrayType.element_shell_stress]
            >>> shell_stress.shape
            (34, 7463, 3, 6)
            >>> # select only parts from part_ids
            >>> shell_stress_parts = shell_stress[:, mask]
        """

        # nodes are treated separately
        if filter_type == FilterType.NODE:
            node_index_arrays = []

            if ArrayType.element_shell_node_indexes in self.arrays:
                shell_filter = self.get_part_filter(
                    FilterType.SHELL, part_ids, for_state_array=False
                )
                shell_node_indexes = self.arrays[ArrayType.element_shell_node_indexes]
                node_index_arrays.append(shell_node_indexes[shell_filter].flatten())

            if ArrayType.element_solid_node_indexes in self.arrays:
                solid_filter = self.get_part_filter(
                    FilterType.SOLID, part_ids, for_state_array=False
                )
                solid_node_indexes = self.arrays[ArrayType.element_solid_node_indexes]
                node_index_arrays.append(solid_node_indexes[solid_filter].flatten())

            if ArrayType.element_tshell_node_indexes in self.arrays:
                tshell_filter = self.get_part_filter(
                    FilterType.TSHELL, part_ids, for_state_array=False
                )
                tshell_node_indexes = self.arrays[ArrayType.element_tshell_node_indexes]
                node_index_arrays.append(tshell_node_indexes[tshell_filter].flatten())

            return np.unique(np.concatenate(node_index_arrays))

        # we need part ids first
        if ArrayType.part_ids in self.arrays:
            d3plot_part_ids = self.arrays[ArrayType.part_ids]
        elif ArrayType.part_titles_ids in self.arrays:
            d3plot_part_ids = self.arrays[ArrayType.part_titles_ids]
        else:
            msg = "D3plot does neither contain '%s' nor '%s'"
            raise RuntimeError(msg, ArrayType.part_ids, ArrayType.part_titles_ids)

        # if we filter parts we can stop here
        if filter_type == FilterType.PART:
            return np.isin(d3plot_part_ids, part_ids)

        # get part indexes from part ids
        part_indexes = np.argwhere(np.isin(d3plot_part_ids, part_ids)).flatten()

        # associate part indexes with entities
        if filter_type == FilterType.BEAM:
            entity_part_indexes = self.arrays[ArrayType.element_beam_part_indexes]
        elif filter_type == FilterType.SHELL:
            entity_part_indexes = self.arrays[ArrayType.element_shell_part_indexes]
            # shells may contain "rigid body shell elements"
            # for these shells no state data is output and thus
            # the state arrays have a reduced element count
            if for_state_array and self._material_section_info.n_rigid_shells != 0:
                mat_types = self.arrays[ArrayType.part_material_type]
                mat_type_filter = mat_types[entity_part_indexes] != 20
                entity_part_indexes = entity_part_indexes[mat_type_filter]

        elif filter_type == FilterType.TSHELL:
            entity_part_indexes = self.arrays[ArrayType.element_tshell_part_indexes]
        elif filter_type == FilterType.SOLID:
            entity_part_indexes = self.arrays[ArrayType.element_solid_part_indexes]
        else:
            msg = "Invalid filter_type '%s'. Use lasso.dyna.FilterType."
            raise ValueError(msg, filter_type)
        mask = np.isin(entity_part_indexes, part_indexes)
        return mask

if __name__=='__main__':
    d = D3plot('./d3plot', state_filter={-1})
    print('Объемных элементов: ', d.header.n_solids)
    print('Узлов: ', d.header.n_nodes)
    # d.header.print()
    print('Число шагов: ', d.n_timesteps)
    print(d.arrays['timesteps'])
    print(d.arrays['element_solid_effective_plastic_strain'][-1][0][0])