import ducc0
import numpy as _np

# setup a few required variables
if ducc0 is not None:
    import os as _os

    try:
        nthreads = int(_os.environ["OMP_NUM_THREADS"])
    except:
        nthreads = 0


def set_nthreads(ntnew):
    global nthreads
    nthreads = ntnew

def _ralm2cilm(alm, lmax):
    cilm = _np.zeros((2, lmax + 1, lmax + 1), dtype=_np.float64)
    cilm[0, :, 0] = alm[0:lmax + 1].real
    ofs = lmax + 1
    for m in range(1, lmax + 1):
        cilm[0, m:, m] = alm[ofs:ofs + lmax + 1 - m].real
        cilm[1, m:, m] = alm[ofs:ofs + lmax + 1 - m].imag
        ofs += lmax + 1 - m
    return cilm
    
def _get_norm(lmax, norm):
    if norm == 1:
        return _np.full(lmax + 1, _np.sqrt(4 * _np.pi))
    if norm == 2:
        return _np.sqrt(4 * _np.pi / (2 * _np.arange(lmax + 1) + 1.0))
    if norm == 3:
        return _np.sqrt(2 * _np.pi / (2 * _np.arange(lmax + 1) + 1.0))
    if norm == 4:
        return _np.ones(lmax + 1)
    raise RuntimeError("unsupported normalization")

def _apply_norm(alm, lmax, norm, csphase, reverse):
    lnorm = _get_norm(lmax, norm)
    if reverse:
        lnorm = 1.0 / lnorm
    alm[0:lmax + 1] *= lnorm[0:lmax + 1]
    lnorm *= _np.sqrt(2.0) if reverse else (1.0 / _np.sqrt(2.0))
    mlnorm = -lnorm
    ofs = lmax + 1
    for m in range(1, lmax + 1):
        if csphase == 1:
            if m & 1:
                alm[ofs:ofs + lmax + 1 - m].real *= mlnorm[m:]
                alm[ofs:ofs + lmax + 1 - m].imag *= lnorm[m:]
            else:
                alm[ofs:ofs + lmax + 1 - m].real *= lnorm[m:]
                alm[ofs:ofs + lmax + 1 - m].imag *= mlnorm[m:]
        else:
            alm[ofs:ofs + lmax + 1 - m].real *= lnorm[m:]
            alm[ofs:ofs + lmax + 1 - m].imag *= mlnorm[m:]
        ofs += lmax + 1 - m
    if norm == 3:  # special treatment for unnormalized a_lm
        r = _np.arange(lmax + 1)
        fct = _np.ones(lmax + 1)
        ofs = lmax + 1
        if reverse:
            alm[0:lmax + 1] /= _np.sqrt(2)
            for m in range(1, lmax + 1):
                fct[m:] *= _np.sqrt((r[m:] + m) * (r[m:] - m + 1))
                alm[ofs:ofs + lmax + 1 - m] /= fct[m:]
                ofs += lmax + 1 - m
        else:
            alm[0:lmax + 1] *= _np.sqrt(2)
            for m in range(1, lmax + 1):
                fct[m:] *= _np.sqrt((r[m:] + m) * (r[m:] - m + 1))
                alm[ofs:ofs + lmax + 1 - m] *= fct[m:]
                ofs += lmax + 1 - m
    return alm


def _extract_alm(alm, lmax, norm, csphase):
    _apply_norm(alm, lmax, norm, csphase, True)
    return _ralm2cilm(alm, lmax)

def _analyze_DH_adjoint(map, lmax):
    alm = ducc0.sht.experimental.adjoint_synthesis_2d(
        map=map.reshape((1, map.shape[0], map.shape[1])),
        spin=0,
        lmax=lmax,
        geometry="DH",
        nthreads=nthreads,
    )
    return alm[0]

def _analyze_DH(map, lmax):
    alm = ducc0.sht.experimental.analysis_2d(
        map=map.reshape((1, map.shape[0], map.shape[1])),
        spin=0,
        lmax=lmax,
        geometry="DH",
        nthreads=nthreads,
    )
    return alm[0]

def SHExpandDH(grid, norm=1, sampling=1, csphase=1, lmax_calc=None, flag = False):
    if grid.shape[1] != sampling * grid.shape[0]:
        raise RuntimeError("grid resolution mismatch")
    if lmax_calc is None:
        lmax_calc = grid.shape[0] // 2 - 1
    if lmax_calc > (grid.shape[0] // 2 - 1):
        raise RuntimeError("lmax_calc too high")
    if flag:
        alm = _analyze_DH_adjoint(grid, lmax_calc)
    else:
        alm = _analyze_DH(grid, lmax_calc)
    return _extract_alm(alm, lmax_calc, norm, csphase)