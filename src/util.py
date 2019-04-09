import os
import shutil
import logging
import numpy as num
from pyrocko import orthodrome

logger = logging.getLogger('pinky.util')


def ensure_dir(p):
    if not os.path.exists(p):
        os.makedirs(p)


class ListEmpty(Exception):
    pass


def ensure_list(l):
    if not len(l):
        raise ListEmpty()


def first_element(x):
    if len(x) <= 1:
        return x[0]
    elif len(x) > 1:
        raise Exception('%s has more than one item' % x)


def filter_oob(sources, targets, config):
    '''Filter sources that will be out of bounds of GF database.'''
    nsources, ntargets = len(sources), len(targets)
    slats, slons = num.empty(nsources), num.empty(nsources)
    sdepth = num.empty(nsources)
    tlats, tlons = num.empty(ntargets), num.empty(ntargets)
    televations = num.empty(ntargets)

    for i_s, s in enumerate(sources):
        slats[i_s], slons[i_s], sdepth[i_s] = *s.effective_latlon, s.depth

    for i_t, t in enumerate(targets):
        tlats[i_t], tlons[i_t] = t.effective_latlon
        televations[i_t] = t.elevation

    # dmax = config.distance_max
    # nsources = len(sources)
    # sources_out = []
    # for i_s, s in enumerate(sources[::-1]):
    #     for t in targets:
    #         print(t.distance_to(s))
    #         if t.distance_to(s) > dmax:
    #             print('break')
    #             break
    #     else:
    #         print('fine')
    #         sources_out.append(s)

    # return sources_out

    dists = num.empty((ntargets, nsources))
    depths = num.empty((ntargets, nsources))
    for i in range(ntargets):
        dists[i] = orthodrome.distance_accurate50m_numpy(
                slats, slons, tlats[i], tlons[i])
        depths[i] = sdepth + televations[i]

    i_dist = num.where(num.any(num.logical_or(
            dists > config.distance_max-100, dists < config.distance_min+100), axis=0))[0]
    i_depth = num.where(num.any(num.logical_or(
            depths > config.source_depth_max-100, depths < config.source_depth_min+100), axis=0))[0]
    print(i_depth)
    i_filter = num.union1d(i_depth, i_dist)

    print(dists)
    logger.warn('Removing %i / %i sources which would be out of bounds' %
            (len(i_filter), nsources))

    for i in i_filter[::-1]:
        # print('.'*10)
        # for t in targets:
        #     print(sources[i].distance_to(t))
        sources.pop(i)
        # del sources[i]
    _d = [t.distance_to(s) for t in targets for s in sources]
    print(max(_d))
    print(min(_d))
    _d = [s.depth+t.elevation for t in targets for s in sources]
    print(min(_d))
    print(max(_d))
    return sources

def delete_if_exists(dir_or_file):
    '''Deletes `dir_or_file` if exists'''
    if os.path.exists(dir_or_file):
        if os.path.isfile(dir_or_file):
            logger.debug('deleting file: %s' % dir_or_file)
            os.remove(dir_or_file)
        else:
            logger.debug('deleting directory: %s' % dir_or_file)
            shutil.rmtree(dir_or_file)


def nsl(tr):
    return tr.nslc_id[:3]


def append_to_dict(d, k, v):
    _m = d.get(k, [])
    _m.append(v)
    d[k] = _m


def snr(chunk, split_factor):
    isplit = int(chunk.shape[1] * split_factor)
    chunk_masked = num.ma.masked_invalid(chunk)
    chunk = chunk**2
    sig = chunk[:, :isplit]
    noise = chunk[:, isplit:]
    s = num.sqrt(num.sum(sig, axis=1))
    n = num.sqrt(num.sum(noise, axis=1))

    return num.nanmean(s/n) / split_factor


def list2string(l, fill=', '):
    """
    Convert list of string to single string.
    copied from BEAT

    Parameters
    ----------
    l: list
        of strings
    """
    return fill.join('"%s"' % listentry for listentry in l)


def get_rotation_matrix(axes=['x', 'y', 'z']):
    """
    Return a function for 3-d rotation matrix for a specified axis.
    copied from BEAT

    Parameters
    ----------
    axes : str or list of str
        x, y or z for the axis

    Returns
    -------
    func that takes an angle [rad]
    """
    ax_avail = ['x', 'y', 'z']
    for ax in axes:
        if ax not in ax_avail:
            raise TypeError(
                'Rotation axis %s not supported!'
                ' Available axes: %s' % (ax, list2string(ax_avail)))

    def rotx(angle):
        cos_angle = num.cos(angle)
        sin_angle = num.sin(angle)
        return num.array(
            [[1, 0, 0],
             [0, cos_angle, -sin_angle],
             [0, sin_angle, cos_angle]], dtype='float64')

    def roty(angle):
        cos_angle = num.cos(angle)
        sin_angle = num.sin(angle)
        return num.array(
            [[cos_angle, 0, sin_angle],
             [0, 1, 0],
             [-sin_angle, 0, cos_angle]], dtype='float64')

    def rotz(angle):
        cos_angle = num.cos(angle)
        sin_angle = num.sin(angle)
        return num.array(
            [[cos_angle, -sin_angle, 0],
             [sin_angle, cos_angle, 0],
             [0, 0, 1]], dtype='float64')

    R = {'x': rotx,
         'y': roty,
         'z': rotz}

    if isinstance(axes, list):
        return R
    elif isinstance(axes, str):
        return R[axes]
    else:
        raise Exception('axis has to be either string or list of strings!')

