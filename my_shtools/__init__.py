"""
pyshtools subpackage that includes all Python wrapped Fortran routines.
"""
import os as _os
import numpy as _np

# Import all wrapped SHTOOLS functions

# legendre
from pyshtools._SHTOOLS import PlmBar
from pyshtools._SHTOOLS import PlmBar_d1
from pyshtools._SHTOOLS import PlBar
from pyshtools._SHTOOLS import PlBar_d1
from pyshtools._SHTOOLS import PlmON
from pyshtools._SHTOOLS import PlmON_d1
from pyshtools._SHTOOLS import PlON
from pyshtools._SHTOOLS import PlON_d1
from pyshtools._SHTOOLS import PlmSchmidt
from pyshtools._SHTOOLS import PlmSchmidt_d1
from pyshtools._SHTOOLS import PlSchmidt
from pyshtools._SHTOOLS import PlSchmidt_d1
from pyshtools._SHTOOLS import PLegendreA
from pyshtools._SHTOOLS import PLegendreA_d1
from pyshtools._SHTOOLS import PLegendre
from pyshtools._SHTOOLS import PLegendre_d1

# expand
from pyshtools._SHTOOLS import SHExpandDH
from pyshtools._SHTOOLS import MakeGridDH
from pyshtools._SHTOOLS import SHExpandDHC
from pyshtools._SHTOOLS import MakeGridDHC
from pyshtools._SHTOOLS import SHGLQ
from pyshtools._SHTOOLS import SHExpandGLQ
from pyshtools._SHTOOLS import MakeGridGLQ
from pyshtools._SHTOOLS import SHExpandGLQC
from pyshtools._SHTOOLS import MakeGridGLQC
from pyshtools._SHTOOLS import GLQGridCoord
from pyshtools._SHTOOLS import SHExpandLSQ
from pyshtools._SHTOOLS import SHExpandWLSQ
from pyshtools._SHTOOLS import MakeGrid2D
from pyshtools._SHTOOLS import MakeGridPoint
from pyshtools._SHTOOLS import MakeGridPointC
from pyshtools._SHTOOLS import SHMultiply
from pyshtools._SHTOOLS import MakeGradientDH
from my_backends.ducc0_wrapper import MakeGridDH_adjoint_analysis

# shio
from pyshtools._SHTOOLS import SHRead2
from pyshtools._SHTOOLS import SHRead2Error
from pyshtools._SHTOOLS import SHReadJPL
from pyshtools._SHTOOLS import SHReadJPLError
from pyshtools._SHTOOLS import SHCilmToCindex
from pyshtools._SHTOOLS import SHCindexToCilm
from pyshtools._SHTOOLS import SHCilmToVector
from pyshtools._SHTOOLS import SHVectorToCilm
from pyshtools._SHTOOLS import SHrtoc
from pyshtools._SHTOOLS import SHctor

# spectralanalysis
from pyshtools._SHTOOLS import SHAdmitCorr
from pyshtools._SHTOOLS import SHConfidence
from pyshtools._SHTOOLS import SHMultiTaperSE
from pyshtools._SHTOOLS import SHMultiTaperCSE
from pyshtools._SHTOOLS import SHLocalizedAdmitCorr
from pyshtools._SHTOOLS import SHReturnTapers
from pyshtools._SHTOOLS import SHReturnTapersM
from pyshtools._SHTOOLS import ComputeDm
from pyshtools._SHTOOLS import ComputeDG82
from pyshtools._SHTOOLS import SHFindLWin
from pyshtools._SHTOOLS import SHBiasK
from pyshtools._SHTOOLS import SHMTCouplingMatrix
from pyshtools._SHTOOLS import SHBiasAdmitCorr
from pyshtools._SHTOOLS import SHMTDebias
from pyshtools._SHTOOLS import SHMTVarOpt
from pyshtools._SHTOOLS import SHMTVar
from pyshtools._SHTOOLS import SHSjkPG
from pyshtools._SHTOOLS import SHMultiTaperMaskSE
from pyshtools._SHTOOLS import SHMultiTaperMaskCSE
from pyshtools._SHTOOLS import SHReturnTapersMap
from pyshtools._SHTOOLS import SHBiasKMask
from pyshtools._SHTOOLS import ComputeDMap
from pyshtools._SHTOOLS import Curve2Mask
from pyshtools._SHTOOLS import SHBias
from pyshtools._SHTOOLS import SphericalCapCoef
from pyshtools._SHTOOLS import SHRotateTapers
from pyshtools._SHTOOLS import SlepianCoeffs
from pyshtools._SHTOOLS import SlepianCoeffsToSH
from pyshtools._SHTOOLS import SHSCouplingMatrix
from pyshtools._SHTOOLS import SHSlepianVar
from pyshtools._SHTOOLS import SHSCouplingMatrixCap

# rotate
from pyshtools._SHTOOLS import djpi2
from pyshtools._SHTOOLS import SHRotateCoef
from pyshtools._SHTOOLS import SHRotateRealCoef

# gravmag
from pyshtools._SHTOOLS import MakeGravGridDH
from pyshtools._SHTOOLS import MakeGravGridPoint
from pyshtools._SHTOOLS import MakeGravGradGridDH
from pyshtools._SHTOOLS import MakeGeoidGridDH
from pyshtools._SHTOOLS import CilmPlusDH
from pyshtools._SHTOOLS import CilmMinusDH
from pyshtools._SHTOOLS import CilmPlusRhoHDH
from pyshtools._SHTOOLS import CilmMinusRhoHDH
from pyshtools._SHTOOLS import BAtoHilmDH
from pyshtools._SHTOOLS import BAtoHilmRhoHDH
from pyshtools._SHTOOLS import DownContFilterMA
from pyshtools._SHTOOLS import DownContFilterMC
from pyshtools._SHTOOLS import NormalGravity
from pyshtools._SHTOOLS import MakeMagGridDH
from pyshtools._SHTOOLS import MakeMagGridPoint
from pyshtools._SHTOOLS import MakeMagGradGridDH

# utils
from pyshtools._SHTOOLS import MakeCircleCoord
from pyshtools._SHTOOLS import MakeEllipseCoord
from pyshtools._SHTOOLS import Wigner3j
from pyshtools._SHTOOLS import DHaj

__all__ = ['PlmBar', 'PlmBar_d1', 'PlBar', 'PlBar_d1', 'PlmON', 'PlmON_d1',
           'PlON', 'PlON_d1', 'PlmSchmidt', 'PlmSchmidt_d1', 'PlSchmidt',
           'PlSchmidt_d1', 'PLegendreA', 'PLegendreA_d1', 'PLegendre',
           'PLegendre_d1', 'SHExpandDH', 'MakeGridDH', 'SHExpandDHC',
           'MakeGridDHC', 'SHGLQ', 'SHExpandGLQ', 'MakeGridGLQ',
           'SHExpandGLQC', 'MakeGridGLQC', 'GLQGridCoord', 'SHExpandLSQ',
           'SHExpandWLSQ', 'MakeGrid2D', 'MakeGridPoint', 'MakeGridPointC',
           'SHMultiply', 'SHRead2', 'SHRead2Error', 'SHReadJPL',
           'SHReadJPLError', 'SHCilmToVector', 'SHVectorToCilm',
           'SHCilmToCindex', 'SHCindexToCilm', 'SHrtoc', 'SHctor',
           'SHAdmitCorr', 'SHConfidence', 'SHMultiTaperSE', 'SHMultiTaperCSE',
           'SHLocalizedAdmitCorr', 'SHReturnTapers', 'SHReturnTapersM',
           'ComputeDm', 'ComputeDG82', 'SHFindLWin', 'SHBiasK',
           'SHMTCouplingMatrix', 'SHBiasAdmitCorr', 'SHMTDebias', 'SHMTVarOpt',
           'SHSjkPG', 'SHMultiTaperMaskSE', 'SHMultiTaperMaskCSE',
           'SHReturnTapersMap', 'SHBiasKMask', 'ComputeDMap', 'Curve2Mask',
           'SHBias', 'SphericalCapCoef', 'djpi2', 'SHRotateCoef',
           'SHRotateRealCoef', 'MakeGravGridDH', 'MakeGravGradGridDH',
           'MakeGeoidGridDH', 'CilmPlusDH', 'CilmMinusDH', 'CilmPlusRhoHDH',
           'CilmMinusRhoHDH', 'BAtoHilmDH', 'BAtoHilmRhoHDH',
           'DownContFilterMA', 'DownContFilterMC', 'NormalGravity',
           'MakeMagGridDH', 'MakeCircleCoord', 'MakeEllipseCoord', 'Wigner3j',
           'DHaj', 'MakeMagGradGridDH', 'SHRotateTapers', 'SlepianCoeffs',
           'SlepianCoeffsToSH', 'SHSCouplingMatrix', 'SHMTVar', 'SHSlepianVar',
           'SHSCouplingMatrixCap', 'MakeGravGridPoint', 'MakeMagGridPoint',
           'MakeGradientDH']

_fortran_functions = ['MakeGridPoint', 'MakeGridPointC', 'DownContFilterMA',
                      'DownContFilterMC', 'SHFindLWin', 'SHSjkPG',
                      'NormalGravity', 'SHConfidence', 'MakeGravGridPoint',
                      'MakeMagGridPoint']

_fortran_subroutines = list(set(__all__) - set(_fortran_functions))


# -----------------------------------------------------------------------------
#
#   Fill the module doc strings with documentation from external
#   files. The doc files are generated during intitial compilation of
#   pyshtools from md formatted text files.
#
# -----------------------------------------------------------------------------
_pydocfolder = _os.path.abspath(_os.path.join(
                   _os.path.split(_os.path.dirname(__file__))[0], 'doc'))

for _name in __all__:
    try:
        _path = _os.path.join(_pydocfolder, _name.lower() + '.doc')

        with open(_path, 'rb') as _pydocfile:
            _pydoc = _pydocfile.read().decode('utf-8')

        setattr(locals()[_name], '__doc__', _pydoc)

    except IOError as msg:
        print(msg)


# -----------------------------------------------------------------------------
#
#   Check the exit status of Fortran routines, raise exceptions, and
#   strip exitstatus from the Python return values.
#
# -----------------------------------------------------------------------------
class SHToolsError(Exception):
    pass


def _shtools_status_message(status):
    '''
    Determine error message to print when an SHTOOLS Fortran 95 routine exits
    improperly.
    '''
    if (status == 1):
        errmsg = 'Improper dimensions of input array.'
    elif (status == 2):
        errmsg = 'Improper bounds for input variable.'
    elif (status == 3):
        errmsg = 'Error allocating memory.'
    elif (status == 4):
        errmsg = 'File IO error.'
    else:
        errmsg = 'Unhandled Fortran 95 error.'
    return errmsg


def _raise_errors(func):
    def wrapped_func(*args, **kwargs):
        returned_values = func(*args, **kwargs)
        if returned_values[0] != 0:
            raise SHToolsError(_shtools_status_message(returned_values[0]))
        elif len(returned_values) == 2:
            return returned_values[1]
        else:
            return returned_values[1:]
    wrapped_func.__doc__ = func.__doc__
    return wrapped_func


for _func in _fortran_subroutines:
    locals()[_func] = _raise_errors(locals()[_func])


# -----------------------------------------------------------------------------
#
#   Vectorize some of the Fortran functions
#
# -----------------------------------------------------------------------------
MakeGridPoint = _np.vectorize(MakeGridPoint, excluded=[0])
MakeGridPointC = _np.vectorize(MakeGridPointC, excluded=[0])
DownContFilterMA = _np.vectorize(DownContFilterMA)
DownContFilterMC = _np.vectorize(DownContFilterMC)
NormalGravity = _np.vectorize(NormalGravity, excluded=[1, 2, 3, 4])
SHConfidence = _np.vectorize(SHConfidence)
