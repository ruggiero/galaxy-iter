from os import path, remove
from sys import exit

import numpy as np
import numpy.random as nprand
from numpy import cos, sin, pi, arccos, log10, exp, arctan, cosh
from numpy.linalg import norm
from scipy.optimize import brentq
from scipy.spatial import KDTree
from scipy.special import i0, i1, k0, k1
from bisect import bisect_left
from argparse import ArgumentParser as parser
from subprocess import call

from snapwrite import process_input, write_snapshot
from pygadgetreader import *


G = 43007.1


def main():
  init()
  galaxy_data = realize_galaxy(N_temp, first=True)
  final_galaxy = realize_galaxy({'gas': N_gas, 'dm': N_halo, 'disk': N_disk, 'bulge': N_bulge})
  for i in ['gas', 'dm', 'disk', 'bulge']:
    iterate(galaxy_data, i, 100)
    transfer_vels(galaxy_data, final_galaxy, i)
    if i == 'gas':
      transfer_gas_dist(galaxy_data, final_galaxy)
  correct_cm_vel(final_galaxy)
  write_input_file(final_galaxy, output)


def init():
  global M_halo, M_disk, M_bulge, M_gas
  global N_halo, N_disk, N_bulge, N_gas
  global a_halo, a_bulge, Rd, z0
  global N_CORES, output
  global N_temp
  flags = parser(description="Description.")
  flags.add_argument('-o', help='The name of the output file.',
                     metavar="init.dat", default="init.dat")
  flags.add_argument('-cores', help='The number of cores to use.',
                     default=1)

  args = flags.parse_args()
  output = args.o
  N_CORES = int(args.cores)

  if not (path.isfile("header.txt") and path.isfile("galaxy_param.txt")):
    print "header.txt or galaxy_param.txt missing."
    exit(0)

  vars_ = process_input("galaxy_param.txt")
  M_halo, M_disk, M_bulge, M_gas = (float(i[0]) for i in vars_[0:4])
  N_halo, N_disk, N_bulge, N_gas = (float(i[0]) for i in vars_[4:8])
  a_halo, a_bulge, Rd, z0 = (float(i[0]) for i in vars_[8:12])
  #N_temp = {'gas': 1e4, 'dm': 5e4, 'disk': 2e4, 'bulge': 2e4}
  N_temp = {'gas': 1e3, 'dm': 5e3, 'disk': 2e3, 'bulge': 2e3}


def realize_galaxy(Npart, first=False):
  coords = {}
  vels = {}
  coords['gas'] = set_disk_positions(Npart['gas'], 0)
  coords['dm'] =  sample_dehnen(Npart['dm'], M_halo, a_halo)
  coords['disk'] = set_disk_positions(Npart['disk'], z0)
  coords['bulge'] =  sample_dehnen(Npart['bulge'], M_halo, a_halo)
  if first:
    vels['gas'] = np.array([rotation_velocity(i) for i in coords['gas']])
    vels['disk'] = np.array([rotation_velocity(i) for i in coords['disk']])
  else:
    vels['gas'] = np.zeros((Npart['gas'], 3))
    vels['disk'] = np.zeros((Npart['disk'], 3))
  vels['dm'] = np.zeros((Npart['dm'], 3))
  vels['bulge'] = np.zeros((Npart['bulge'], 3))
  return {'pos': coords, 'vel': vels}


def rotation_velocity(pos):
  rho = (pos[0]**2 + pos[1]**2)**0.5
  phi = calculate_phi(pos[0], pos[1])
  y = rho/(2*Rd)
  sigma0 = M_halo / (2*pi*Rd**2)
  speed = (4*pi*G*sigma0*y**2*(i0(y)*k0(y) - i1(y)*k1(y)) +
           (G*M_halo*rho)/(rho+a_halo)**2 + (G*M_bulge*rho)/(rho+a_bulge)**2)**0.5
  return (-speed*sin(phi), speed*cos(phi), 0)


# c is the name of the component being iterated, and iterations
# is the number of iterations to perform
def iterate(galaxy_data, c, iterations):
  newsnap = 'temp/snapshot_000'
  program_name = {'gas': 'gadget_0', 'dm': 'gadget_1', 'disk': 'gadget_2', 'bulge': 'gadget_3'}[c]
  new_coords = {}
  new_vels = {}
  for i in range(iterations):
    if c == 'gas':
      write_input_file(galaxy_data, 'temp.dat', null_gas_ids=True)
    else:
      write_input_file(galaxy_data, 'temp.dat')
    call(["mpirun -n %d %s iterate.param" % (N_CORES, program_name)], shell=True)
    new_coords[c] = readsnap(newsnap, 'pos', c)
    new_vels[c] = readsnap(newsnap, 'vel', c)
    call(['rm temp.dat temp/*'], shell=True)
    new_data = {'pos': new_coords, 'vel': new_vels}
    correct_cm_vel(new_data)
    galaxy_data = realize_galaxy(N_temp)
    transfer_vels(new_data, galaxy_data, c)
    if c == 'gas':
      transfer_gas_dist(new_data, galaxy_data)
  return new_data


def correct_cm_vel(galaxy_data):
  coords = np.concatenate((galaxy_data['pos'].values()))
  velcm = np.sum(coords) / len(coords)
  for key in galaxy_data['vel']:
    galaxy_data['vel'][key] -= velcm


# Transfers the velocity using cylindrical coordinates
def transfer_vels(new_data, old_data, c):
  # Number of times each particle in the new model has been used
  N_used = np.zeros(N_temp[c])
  rhos = (new_data['pos'][c][:,0]**2 + new_data['pos'][c][:,1]**2)**0.5
  cyl_data = np.dstack((rhos, abs(new_data['pos'][c][:,2])))[0]
  tree = KDTree(cyl_data)
  for j, p in enumerate(old_data['pos'][c]):
    b = find_best_neighbor(((p[0]**2+p[1]**2)**0.5, abs(p[2])), tree, N_used)
    N_used[b] += 1

    x, y, z = p
    nx, ny, nz = new_data['pos'][c][b]
    vx, vy, vz = old_data['vel'][c][j]
    nvx, nvy, nvz = new_data['vel'][c][b]

    phi = calculate_phi(x, y)
    nphi = calculate_phi(nx, ny)
    if c == 'gas':
      nvr = 0
    else:
      nvr = nvy*sin(nphi) + nvx*cos(nphi)
    nvphi = nvy*cos(nphi) - nvx*sin(nphi)
 
    old_data['vel'][c][j] = (nvr*cos(phi) - nvphi*sin(phi), nvr*sin(phi) + nvphi*cos(phi), nvz)


def transfer_gas_dist(new_data, old_data):
  N_used = np.zeros(N_temp['gas'])
  rhos = (new_data['pos']['gas'][:,0]**2 + new_data['pos']['gas'][:,1]**2)**0.5
  tree = KDTree(rhos[:,None])
  for j, p in enumerate(old_data['pos']['gas']):
    b = find_best_neighbor([(p[0]**2+p[1]**2)**0.5], tree, N_used)
    N_used[b] += 1
    old_data['pos']['gas'][j][2] = new_data['pos']['gas'][b][2]


def find_best_neighbor(pos, tree, N_used):
  closest = tree.query(pos, 10)
  best = closest[1][0]
  distance = closest[0][0]
  # Iterating over the indexes of the closest neighbors
  for i, p in enumerate(closest[1][1:]): 
    if N_used[p] < N_used[best]:
      best = p
      distance = closest[0][i]
    elif N_used[p] == N_used[best] and closest[0][i] < distance:
      best = p
      distance = closest[0][i]
  return best    


def calculate_phi(x, y):
  if(x > 0 and y > 0):
    phi = arctan(y/x)
  elif(x < 0 and y > 0):
    phi = pi - arctan(-y/x)
  elif(x < 0 and y < 0):
    phi = pi + arctan(y/x)
  elif(x > 0 and y < 0):
    phi = 2 * pi - arctan(-y/x)
  return phi


def dehnen_inverse_cumulative(Mc, M, a, core):
  if(core):
    return ((a * (Mc**(2/3.)*M**(4/3.) + Mc*M + Mc**(4/3.)*M**(2/3.))) /
            (Mc**(1/3.) * M**(2/3.) * (M-Mc)))
  else:
    return (a * ((Mc*M)**0.5 + Mc)) / (M-Mc)


def sample_dehnen(N, M, a, core=False):
  # The factor M * 200^2 / 201^2 restricts the radius to 200 * a.
  radii = dehnen_inverse_cumulative(nprand.sample(N) *
    ((M*40000) / 40401), M, a, core)
  thetas = np.arccos(nprand.sample(N)*2 - 1)
  phis = 2 * pi * nprand.sample(N)
  xs = radii * sin(thetas) * cos(phis)
  ys = radii * sin(thetas) * sin(phis)
  zs = radii * cos(thetas)
  coords = np.column_stack((xs, ys, zs))
  return coords


def set_disk_positions(N, z0):
  radii = np.zeros(N)
  # The maximum radius is restricted to 60 kpc.
  sample = nprand.sample(N) * disk_radial_cumulative(60)
  for i, s in enumerate(sample):
    radii[i] = disk_radial_inverse_cumulative(s)
  if(z0 > 0):
    zs = disk_height_inverse_cumulative(nprand.sample(N), z0)
  else:
    zs = np.zeros(N)
  phis = 2 * pi * nprand.sample(N)
  xs = radii * cos(phis)
  ys = radii * sin(phis)
  coords = np.column_stack((xs, ys, zs))
  return coords


def disk_radial_cumulative(r):
  return (Rd**2-(Rd**2+r*Rd)*exp(-r/Rd))/Rd**2


# 'frac' is a number between 0 and 1.
def disk_radial_inverse_cumulative(frac):
  return brentq(lambda r: disk_radial_cumulative(r) - frac, 0, 1.0e10)


def disk_height_inverse_cumulative(frac, z0):
  return 0.5 * z0 * np.log(frac/(1-frac))


# Npart...
def write_input_file(galaxy_data, name, null_gas_ids=False):
  Npart = [len(galaxy_data['pos']['gas']), len(galaxy_data['pos']['dm']),
    len(galaxy_data['pos']['disk']), len(galaxy_data['pos']['bulge']), 0, 0]
  coords = np.concatenate((galaxy_data['pos']['gas'], galaxy_data['pos']['dm'],
    galaxy_data['pos']['disk'], galaxy_data['pos']['bulge']))
  coords.shape = (1, -1) # Linearizing the array.
  vels = np.concatenate((galaxy_data['vel']['gas'], galaxy_data['vel']['dm'],
    galaxy_data['vel']['disk'], galaxy_data['vel']['bulge']))
  vels.shape = (1, -1) # Linearizing the array.
  if null_gas_ids:
    ids = np.concatenate((np.zeros(Npart[0]), np.arange(Npart[0]+1, sum(Npart)+1, 1)))
  else:
    ids = np.arange(1, sum(Npart)+1, 1)
  m_gas = np.empty(Npart[0])
  m_gas.fill(M_gas/Npart[0])
  m_halo = np.empty(Npart[1])
  m_halo.fill(M_halo/Npart[1])
  m_disk = np.empty(Npart[2])
  m_disk.fill(M_disk/Npart[2])
  m_bulge = np.empty(Npart[3])
  m_bulge.fill(M_disk/Npart[3])
  masses = np.concatenate((m_gas, m_halo, m_disk, m_bulge))
  U = np.zeros(Npart[0])
  rhos = np.zeros(Npart[0])
  smooths = np.zeros(Npart[0])
  write_snapshot(n_part=Npart, outfile=name,
    data_list=[coords[0], vels[0], ids, masses, U, rhos, smooths])


if __name__ == '__main__':
  main()
