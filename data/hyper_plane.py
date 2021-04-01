import numpy as np
import os
import torch
from itertools import combinations_with_replacement, permutations

from torch.utils.data import Dataset


class HyperPlaneDataset(Dataset):
    def __init__(self, num_points, dim, flip_axes=False):
        self.num_points = num_points
        self.dim = dim
        self.flip_axes = flip_axes
        self.data = None
        self.reset()

    def __getitem__(self, item):
        return self.data[item]

    def __len__(self):
        return self.num_points

    # TODO: Need to change this to permute axes
    def reset(self):
        self._create_data()
        if self.flip_axes:
            x1 = self.data[:, 0]
            x2 = self.data[:, 1]
            self.data = torch.stack([x2, x1]).t()

    def _create_data(self):
        raise NotImplementedError

    def sample(self, num):
        # Expect a list for the num
        self.num_points = num[0]
        self._create_data()
        return self.data


class SparseHyperCheckerboardDataset(HyperPlaneDataset):
    """
    This class generates an object that has a checkerboard in every consecutive 2D projection. It is sparse in the sense
    that the number of hyper checkers it contains is sparse - the density per hyper checker will be higher than the
    class below. With each extra dimension added the number of checkers doubles.
    """

    def __init__(self, num_points, dim, flip_axes=False):
        super(SparseHyperCheckerboardDataset, self).__init__(num_points, dim, flip_axes=flip_axes)

    def append_axis(self, x):
        xi_ = torch.rand(self.num_points) - torch.randint(0, 2, [self.num_points]).float() * 2
        return xi_ + torch.floor(x) % 2

    def _create_data(self):
        self.bounded = True
        if self.dim > 1:
            x1 = torch.rand(self.num_points) * 4 - 2
            axis_list = [x1]
            for i in range(1, self.dim):
                axis_list += [self.append_axis(axis_list[-1])]
            self.data = torch.stack(axis_list).t() * 2
        else:
            self.data = torch.cat(
                (torch.rand(int(self.num_points / 2)) * 2 - 4, torch.rand(int(self.num_points / 2)) * 2))


class HyperCheckerboardDataset(HyperPlaneDataset):
    """
    This class generates an object that is truly a checkerboard. With each dimension that is added the number of
    checkers qudruples.
    """

    def __init__(self, num_points, dim, flip_axes=False):
        super(HyperCheckerboardDataset, self).__init__(num_points, dim, flip_axes=flip_axes)

    def make_cube(self):
        return torch.rand(self.dim, self.num_points)

    @staticmethod
    def count_oob(cube):
        """
        Get the fraction of samples outside of the bounds of the cube
        """
        out_range = (cube > 4).any(1) | (cube < -4).any(1)
        out_range = out_range.sum() / cube.shape[0]
        return out_range

    @staticmethod
    def count_ood(cube):
        """
        :param cube: A tensor of samples from a cube.shape[1] dimensional space.
        :return: The fraction of samples that are within a hypercheckerboard
        """
        dim = cube.shape[1]
        # Get the cube assignments for each axis
        labels = ((cube + 4) / 2).floor() % 2
        # If the sum is odd and so is the dimension then the point is in the checkerboard, and the same for even
        mx = labels.sum(1) % (2 + dim % 2)
        # We also need to set all points outside of the data range to one in mx
        out_range = out_range = (cube > 4).any(1) | (cube < -4).any(1)
        return ((out_range.type(mx.dtype) + mx) > 0).sum() / len(mx)

    @staticmethod
    def split_cube(cube):
        """
        An n-dimensional checkerboard is just a set of 2D checkerboards, so all that is required is to find the correct
        shift in a two dimensional plane. This is defined by a set of oscillating transformations depending on the value
        of the nth coordinates.
        :param cube: an n-dimensional cube with values uniformly distributed in (0, 1)
        :return: an n-dimensional checkerboard
        """
        # Split first axis
        ax0 = cube[0]
        ax0 -= 0.5
        ax0[ax0 < 0] = ax0[ax0 < 0] * 2 - 1
        ax0[ax0 > 0] = ax0[ax0 > 0] * 2
        if cube.shape[0] > 1:
            # Scale other axes to be in a useful range for floor divide
            cube[1:] = cube[1:] * 4
            # Define the shifts
            displace = cube[1:].floor() % 2
            shift = displace[0]
            # We need an algebra that satisies: 1 * 0 = 0, 1 * 1 = 1, 0 * 1 = 0, 0 * 0 = 1
            # This is achieved with * = (==)
            for ax in displace[1:]:
                shift = shift == ax
            ax0 += shift
            cube[1:] -= 2
        cube *= 2
        return cube.t()

    def _create_data(self):
        self.bounded = True
        # All checkerboards start from an N dim checkerboard in [0, 1] #TODO: there is a slight discrepancy here with the inverse Box-Muller transform
        cube = self.make_cube()
        self.data = self.split_cube(cube)


class HyperSpheres(HyperPlaneDataset):
    def __init__(self, num_points, dim, flip_axes=False):
        if num_points % 4 != 0:
            raise ValueError('Number of data points must be a multiple of four')
        # Calculating all of these permutations can be expensive in high dimensions, so set a cap at 6 dims
        nperm = dim if dim <= 6 else 6
        combs = combinations_with_replacement((-1, 1), nperm)
        perms = [list(permutations(t, nperm)) for t in combs]
        perms = [list(item) for sublist in perms for item in sublist]
        self.centers = np.unique(perms, axis=0)
        super().__init__(num_points, dim, flip_axes)
        self.bounded = True

    def create_sphere(self, num_per_circle, std=0.1):
        angles = np.pi * torch.rand(num_per_circle * (self.dim - 1)).view(self.dim - 1, num_per_circle)
        angles[-1] *= 2
        data = torch.ones((self.dim, num_per_circle))
        j = 1
        for coord in range(self.dim - 1):
            for i, u in enumerate(angles[:j]):
                if i == (j - 1):
                    data[coord] *= torch.cos(u)
                else:
                    data[coord] *= torch.sin(u)
            j += 1

        if angles.shape[0] == 1:
            data[-1] = torch.sin(angles)
        else:
            data[-1] = torch.prod(torch.sin(angles), 0)

        data = 2 * data.t()
        data += std * torch.randn(data.shape)
        return data

    def _create_data(self):
        num_per_circle = self.num_points // 4
        self.data = torch.cat(
            [self.create_sphere(num_per_circle) - torch.Tensor(center)
             for center in self.centers]
        )


def plot_projection(dataset, name, shft=0):
    from matplotlib import pyplot as plt
    from nsf_utils import torchutils
    from utils.io import get_top_dir

    data = torchutils.tensor2numpy(dataset.data)
    fig, ax = plt.subplots(1, 1, figsize=(5, 5))
    bound = 4
    bounds = [[-bound, bound], [-bound, bound]]
    ax.hist2d(data[:, 0 + shft], data[:, 1 + shft], bins=256, range=bounds)
    ax.set_xlim(bounds[0])
    ax.set_ylim(bounds[1])
    fig.savefig(get_top_dir() + '/images/{}.png'.format(name))


def threeDscatter(dataset, nsample, name):
    from mpl_toolkits import mplot3d
    from matplotlib import pyplot as plt
    from utils.io import get_top_dir
    dim = 3
    data = dataset(nsample, dim).data
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(data[:, 0], data[:, 1], data[:, 2], marker='x', alpha=0.01)
    fig.savefig(get_top_dir() + '/images/3D_{}.png'.format(name))


def plot_slices(dataset, dim, nsample, name, shft=3):
    from matplotlib import pyplot as plt
    from utils.io import get_top_dir

    fig, ax = plt.subplots(1, 1, figsize=(5, 5))
    bound = 4
    nbins = 50
    bin_edges = np.linspace(-bound, bound, nbins + 1)
    counts = np.zeros((nbins, nbins))
    it = 0
    max_it = 100
    inds = np.arange(nsample)
    mx = np.ones(dim, dtype='bool')
    mx[[shft, shft+1]] = 0
    while np.sum(counts) < int(1e4) and (it < max_it):
        it += 1
        data = dataset(nsample, dim).data
        # Apply a slice to the data
        # to_slice = data[:, 2:]
        to_slice = data[:, mx]
        mask = torch.all((to_slice > 0) & (to_slice < 2), 1)
        data = data[mask.type(torch.bool)].numpy()
        counts += np.histogram2d(data[:, 0 + shft], data[:, 1 + shft], bins=bin_edges)[0]

    counts[counts == 0] = np.nan
    ax.imshow(counts.T,
              origin='lower', aspect='auto',
              extent=[-bound, bound, -bound, bound],
              )
    fig.savefig(get_top_dir() + '/images/slice_{}.png'.format(name))


def _test():
    # Check all projections are the way they should appear
    dim = 3
    nsample = int(1e6)
    # sparsecheckers = SparseHyperCheckerboardDataset(int(1e6), dim)
    # spheres = HyperSpheres(nsample, dim)
    # for i in range(dim - 1):
    #     plot_projection(sparsecheckers, 'sparse_checkerboard_{}'.format(i), shft=i)
    #     plot_projection(spheres, 'four_circles_{}'.format(i), shft=i)
    #
    # nsample = int(1e5)
    # threeDscatter(HyperCheckerboardDataset, nsample, 'checkerboard')
    # threeDscatter(HyperSpheres, nsample, 'spheres')
    #
    nsample = int(1e6)
    plot_slices(HyperCheckerboardDataset, 10, nsample, 'checkerboard')
    # TODO: use this to inspect the hyperspheres properly and figure out what is the matter.
    plot_slices(HyperSpheres, 5, nsample, 'spheres')

    # Investigate the OOD calculation methods
    dim = 2
    checkers = HyperCheckerboardDataset(nsample, dim)
    print('Testing OOD on only sample. Should be zero, {}'.format(HyperCheckerboardDataset.count_ood(checkers.data)))
    cube = ((checkers.make_cube() * 8) - 4).t()
    print('Testing OOD on cube. Should be one half, {}'.format(HyperCheckerboardDataset.count_ood(cube)))
    print('Testing OOD on half cube. Should be one half, {}'.format(HyperCheckerboardDataset.count_ood((cube + 4) / 2)))
    filled_checkers = torch.cat((checkers.data, cube), 0)
    print('Testing OOD on half cube plus checkers. Should be 0.25, {}'.format(
        HyperCheckerboardDataset.count_ood(filled_checkers)))
    cube[:, 0] += 4
    print('Testing OOD on cube shifted along one dimension. Should be 0.75, {}'.format(
        HyperCheckerboardDataset.count_ood(cube)))
    print('Testing OOB on cube shifted along one dimension. Should be 0.5, {}'.format(
        HyperCheckerboardDataset.count_oob(cube)))


if __name__ == '__main__':
    _test()
