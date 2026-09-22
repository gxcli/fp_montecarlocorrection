''' 
Scripts written by ChatGPT in GitHub Copilot, based on the scripts in the archive folder. 
Modified recently 9/10/2026.
'''
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import concurrent.futures

import parameters as p
import auxiliary_funcs as af 
import collisions_vec as col


def _reflect_velocity_boundaries(v_new): # intermediate, process new velocities to be in boundaries
    v_new = np.asarray(v_new, dtype=float)

    v_new[..., 0] = np.abs(v_new[..., 0])

    xi = v_new[..., 1]
    xi = ((xi + 1) % 4) - 1 # map to periodic domain of length 4 centered at 0
    xi = np.where(xi > 1, 2 - xi, xi) # reflect into [-1,1]

    v_new[..., 1] = xi
    return v_new


def singlestep_mc(v_current, D_func, A_func, dt, R, Phi): # single step of N particles
    v_current = np.asarray(v_current, dtype=float)
    if v_current.ndim == 1:  # original shape (2,0) as [x, xi]
        v_current = v_current.reshape(1, 2)

    D_loc = np.asarray(D_func(v_current), dtype=float) # loc for local # (N, 2, 2)
    A_loc = np.asarray(A_func(v_current), dtype=float) # (N, 2)

    std = np.sqrt(2 * np.stack([D_loc[..., 0, 0], D_loc[..., 1, 1]], axis=-1) * dt) # recall D is diagonal
    dv_D = std * np.random.standard_normal(size=v_current.shape) # N(0, std^2) ~ std * N(0,1)
    dv_A = A_loc * dt

    v_new = v_current + dv_D + dv_A
    v_new = _reflect_velocity_boundaries(v_new)

    lc_condition = np.sqrt(1 - (1 - Phi / v_new[..., 0]**2) / R) - np.abs(v_new[..., 1])
    escape = lc_condition < 0 # boolean array

    if v_current.ndim == 1:
        return v_new[0], bool(escape[0]) # returning [x_new, xi_new], T/F

    return v_new, escape



def _split_source_chunks(source, nprocs):
    source = np.asarray(source, dtype=float)
    if source.ndim == 1:
        source = source.reshape(1, 2)

    numparticles = source.shape[0]
    if nprocs <= 1 or numparticles == 0:
        return [source]

    nprocs = min(int(nprocs), numparticles)
    chunks = np.array_split(source, nprocs)
    return [chunk for chunk in chunks if chunk.size > 0]


def _run_mc_chunk(source, numsteps, D_func, A_func, dt, R, Phi, speed_threshold): 
    source = np.asarray(source, dtype=float)
    if source.ndim == 1:
        source = source.reshape(1, 2)

    numparticles = source.shape[0]
    v_current = source.copy()
    last_velocity = np.zeros_like(v_current)
    step_counts = np.zeros(numparticles, dtype=int)
    escaped = np.zeros(numparticles, dtype=bool)
    if speed_threshold > 0: 
        trapped = np.zeros(numparticles, dtype=bool) # keep track of trapped particles past certain speed
    active = np.ones(numparticles, dtype=bool)

    for step in tqdm(range(numsteps)):
        active_idx = np.nonzero(active)[0] 
        if active_idx.size == 0: # if all terminated or trapped, stop. 
            break

        v_active = v_current[active_idx]
        v_new_active, escape_active = singlestep_mc(v_active, D_func, A_func, dt, R, Phi) # single step

        step_counts[active_idx] += 1
        last_velocity[active_idx] = v_new_active # update state. ditch history of particle though.
        escaped[active_idx] = escape_active
        
        if speed_threshold > 0: 
            trap_active = v_new_active[:, 0] <= speed_threshold
            trapped[active_idx] = trap_active
            stop_active = escape_active | trap_active

        else: 
            stop_active = escape_active

        continue_active = ~stop_active # something wrong here!!!#############################################
        v_current[active_idx[continue_active]] = v_new_active[continue_active]
        active[active_idx[stop_active]] = False

    terminated = escaped

    if speed_threshold > 0: 
        return last_velocity[terminated], step_counts[terminated], escaped[terminated], trapped[terminated]
    else: 
        return last_velocity[terminated], step_counts[terminated], escaped[terminated]


def _run_mc_chunk_wrapper(args):
    return _run_mc_chunk(*args)


def run_mc(source, numsteps, D_func, A_func, dt, R, Phi, speed_threshold=0, nprocs=1):
    source = np.asarray(source, dtype=float)
    chunks = _split_source_chunks(source, nprocs)
    if len(chunks) == 1:
        return _run_mc_chunk(source, numsteps, D_func, A_func, dt, R, Phi, speed_threshold)

    args = [(chunk, numsteps, D_func, A_func, dt, R, Phi, speed_threshold) for chunk in chunks]
    with concurrent.futures.ProcessPoolExecutor(max_workers=len(args)) as executor:
        results = list(executor.map(_run_mc_chunk_wrapper, args))

    if speed_threshold > 0: 
        velocities, steps, escaped, trapped = zip(*results)
        return (
                np.concatenate(velocities, axis=0),
                np.concatenate(steps, axis=0),
                np.concatenate(escaped, axis=0),
                np.concatenate(trapped, axis=0),
            )
    else: 
        velocities, steps, escaped = zip(*results)
        return (
                np.concatenate(velocities, axis=0),
                np.concatenate(steps, axis=0),
                np.concatenate(escaped, axis=0),
            )