import numpy as np
import numba
from pic import initialization as init

def collide_with_neutrals(collided_fraction, T_e, m_neutral, particles_m, particles_v, active_particles):
    collided_count = int(active_particles * collided_fraction)
    collided_indices = np.random.choice(active_particles, collided_count, replace=False)
    neutrals_v = init.maxwell_velocities(T_e, m_neutral, collided_count)
    relative_v = particles_v[collided_indices] - neutrals_v
    cos_Xi = np.sqrt(np.random.random(collided_count)) # for ion-neutral
    if particles_m != m_neutral:                       # for electron-neutral
        cos_Xi = np.sqrt(1 - 2*particles_m/m_neutral * (1 - cos_Xi))
    cos_phi = np.cos(np.random.random(collided_count)*2*np.pi)
    mangle_v_r(relative_v.copy(), cos_Xi, cos_phi)
    particles_v[collided_indices] = relative_v + neutrals_v



@numba.jit(nopython=True)
def perpendicular_vector_normed(vector, ret):
    """Return a normed vector perpendicular to given vector"""
    ret = np.zeros(3)
    if vector[0] == 0 and vector[1] ==0:
        # vector == [0, 0, z]
        ret[0] = 0
        ret[1] = 1
        ret[2] = 0
    else:
        # can deal only with x and y
        ret[0] = - vector[1]
        ret[1] = vector[0]
        ret[2] = 0
        ret /= (ret[0]**2 + ret[1]**2)**0.5
    return ret


@numba.jit(nopython=True)
def mangle_v_r(v_r, cos_Xi, cos_phi):
    e_1 = np.empty(3)
    e_2 = np.empty(3)
    e_3 = np.empty(3)
    for i in range(cos_Xi.shape[0]):
        v_r_norm = np.linalg.norm(v_r[i,:])
        v_new = v_r_norm * cos_Xi[i]
        e_1[:] = v_r[i,:] / v_r_norm
        perpendicular_vector_normed(e_1, e_2)
        perpendicular_vector_normed(e_2, e_3)
        sin_Xi = (1 - cos_Xi[i]**2)**0.5
        sin_phi = (1 - cos_phi[i]**2)**0.5
        v_r[i,:] = v_new * (e_1*cos_Xi[i] + e_2*sin_Xi*cos_phi[i] + e_3*sin_Xi*sin_phi)


