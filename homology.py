from __future__ import division
import numpy as np
import plotly as pt
from scipy.spatial.distance import pdist, squareform
from plotly.graph_objs import Scatter, Layout, Scatter3d, Line, Marker, Data, Figure
from collections import defaultdict

np.random.seed(1000)

class Homology:

    def __init__(self):
        # distance for vietoris rips complex
        self.epsilon = 1.5

        # types of simplicial complex
        self.simplicial_complex_types = ['vietoris_rips']

        # names of datasets
        self.dataset_names_dimensions_types = [('tetrahedron',3, 'simple'), ('circle',2,'cloud'), ('sphere',3,'cloud'),('figure_8',2,'cloud')]

        # number of points used
        self.point_cloud_sizes = [50]

        # data structure used to store various values
        infdict = lambda: defaultdict(infdict)

        """Used to store point cloud datasets.
        Schema:
        for dataset_type == 'simple': self.datasets[dataset_name]
        for dataset_type == 'cloud': self.datasets[dataset_name][point_cloud_size]
        Value Format:
        Datasets are stored as a M x N numpy array where each row is a datapoint. So in an M x N array there are M
        datapoints of dimension N."""
        self.datasets = infdict()

        """Used to store simplicial complexes.
        Schema:
        for dataset_type == 'simple': self.datasets[dataset_name][plex_type]
        for dataset_type == 'cloud': self.datasets[dataset_name][point_cloud_size][plex_type]
        Value Format:
        Simplicial complexes are Python lists containing k-simplices. Each Each k-simplex is a Python list containing
        nonnegative integers.
        Example Value:
        [[0],[1], [2], [0,1],[0,2], [1,2], [0, 1, 2]]"""
        self.simplicial_complexes = infdict()


        """Used to store Betti numbers.
        Schema:
        for dataset_type == 'simple': self.datasets[dataset_name][plex_type]
        for dataset_type == 'cloud': self.datasets[dataset_name][point_cloud_size][plex_type]
        Value Format:
        Each entry is a Python list of non-negative integers with the k-th Betti number located at index k.
        Example Value:
        [2, 1, 1] where Betti_0 = 2, Betti_1 = 1, and Betti_2 = 1
        """
        self.Hk_dims = infdict()

        # for descriptions of these look in compute_homology()
        self.Ck_bases = infdict()
        self.Ck_dims = infdict()
        self.boundary_matrices = infdict()
        self.rr_boundary_mats = infdict()
        self.Bk_bases = infdict()
        self.Bk_dims = infdict()
        self.Zk_bases = infdict()
        self.Zk_dims = infdict()
        self.Hk_bases = infdict()

    def main(self):
        # create datasets, build simplicial complexes, and compute homology
        for dataset_name, dataset_dim, dataset_type in self.dataset_names_dimensions_types:
            for plex_type in self.simplicial_complex_types:
                if dataset_type == 'simple':
                    self.datasets[dataset_name] = self.create_dataset(dataset_name, dataset_type)
                    self.simplicial_complexes[dataset_name][plex_type] = self.create_simplicial_complex(plex_type, dataset_name, dataset_type)
                    self.compute_homology(plex_type, dataset_name, dataset_type)
                    print("DATASET:", dataset_name)
                    print("BETTI NUMBERS:", self.Hk_dims)
                    print("")
                elif dataset_type == 'cloud':
                    for num_points in self.point_cloud_sizes:
                        self.datasets[dataset_name][num_points] = self.create_dataset(dataset_name, dataset_type, num_points=num_points)
                        self.simplicial_complexes[dataset_name][plex_type][num_points] = self.create_simplicial_complex(plex_type, dataset_name, dataset_type, num_points=num_points)
                        self.compute_homology(plex_type, dataset_name, dataset_type, num_points=num_points)
                        if dataset_dim == 2:
                            self.scatter_plot_2d(dataset_name, num_points)
                        elif dataset_dim == 3:
                            self.scatter_plot_3d(dataset_name, num_points)

                        print("DATASET:", dataset_name)
                        print("BETTI NUMBERS:", self.Hk_dims)
                        print("")

    def create_dataset(self, dataset_name, dataset_type, num_points=-1):
        if dataset_type == 'simple':
            if dataset_name == 'tetrahedron':
                mat = np.array([
                    [0,0,0],
                    [0,1,0],
                    [1,0,0],
                    [0,0,1]
                ])
        elif dataset_type == 'cloud':
            if dataset_name == 'circle':
                theta = np.random.uniform(0, 2 * np.pi, (num_points, 1))
                x = np.cos(theta)
                y = np.sin(theta)
                mat = np.hstack((x, y))
            if dataset_name == 'figure_8':
                theta = np.random.uniform(0, 2 * np.pi, (num_points, 1))
                x = np.cos(theta)
                y = np.sin(theta)
                mask = np.random.choice([-1, 1], (num_points, 1))
                mat = np.hstack((x + mask, y))
            if dataset_name == 'sphere':
                theta = np.random.uniform(0, 2 * np.pi, (num_points, 1))
                u = np.random.uniform(-1, 1, (num_points, 1))
                x = np.cos(theta) * np.sqrt(1 - np.square(u))
                y = np.sin(theta) * np.sqrt(1 - np.square(u))
                z = u
                mat = np.hstack((x, y, z))
        return mat

    def create_simplicial_complex(self, plex_type, dataset_name, dataset_type, num_points=-1):
        if plex_type == 'vietoris_rips':
            return self.vietoris_rips_complex(dataset_name, dataset_type, num_points=num_points)

    def vietoris_rips_complex(self, dataset_name, dataset_type, num_points=-1):
        plex = []

        # get data
        if dataset_type == 'simple':
            data = self.datasets[dataset_name]
        elif dataset_type =='cloud':
            data = self.datasets[dataset_name][num_points]

        # get the number of points in the dataset. Must include this line because otherwise 'simple' datasets will have num_points = -1
        num_points = data.shape[0]

        # get the distance matrix of the points in out dataset, shape: (self.num_points, self.num_points)
        # D(i,x) = euclidean distance between point i and point x
        dist_mat = squareform(pdist(data))

        # add vertices
        for point_index in range(num_points):
            plex.append([point_index])

        # returns true if the distance between x and y is less than self.epsilon
        def d_less_than_ep(x,y):
            if dist_mat[x, y] <= self.epsilon:
                return True

        # add edges
        for first_index in range(num_points):
            for second_index in range(first_index + 1, num_points):
                if d_less_than_ep(first_index, second_index):
                    plex.append([first_index, second_index])

        # add faces
        for first_index in range(num_points):
            for second_index in range(first_index + 1, num_points):
                for third_index in range(second_index + 1, num_points):
                    if d_less_than_ep(first_index, second_index) and d_less_than_ep(first_index, third_index) and \
                            d_less_than_ep(second_index, third_index):
                        plex.append([first_index, second_index, third_index])

        # add 3-simplices
        for first_index in range(num_points):
            for second_index in range(first_index + 1, num_points):
                for third_index in range(second_index + 1, num_points):
                    for fourth_index in range(third_index + 1, num_points):
                        if d_less_than_ep(first_index, second_index) and d_less_than_ep(first_index, third_index) and \
                                d_less_than_ep(first_index, fourth_index) and \
                                d_less_than_ep(second_index, third_index) and \
                                d_less_than_ep(second_index, fourth_index) and \
                                d_less_than_ep(third_index, fourth_index):
                            plex.append([first_index, second_index, third_index, fourth_index])

        return plex

    def swap_rows(self, mat, i, j):
        """Swaps the row (i,:) with the row (j,:) in a 2-D numpy array"""
        temp = np.copy(mat[i,:])
        mat[i,:] = mat[j,:]
        mat[j,:] = temp

    def row_reduce(self, mat):
        """Row reduce a single boundary matrix"""

        # only row reduce the matrix if it is nonempty
        if mat.size != 0:
            height = mat.shape[0]
            width = mat.shape[1]

            """downward pass:
                    1) make all pivot values equal to 1
                    2) make all entries below pivot values equal to 0"""

            current_row = 0
            for c in range(width):
                if current_row < height: # check to see if row index is inside the matrix
                    # if possible, make the array entry at (current_row, c) nonzero by swapping the current row with rows below
                    if mat[current_row, c] == 0:
                        for r in range(current_row + 1, height):  # (r, c) starts at (current_row + 1, c) and moves down the column (:, c)
                            if mat[r,c] != 0:
                                self.swap_rows(mat, r, current_row)
                                break

                    if mat[current_row,c] != 0:
                        # make d entry equal to 1 by scaling the row
                        if mat[current_row,c] < 0 or mat[current_row,c] > 1:
                            mat[current_row,:] = mat[current_row,:]/mat[current_row,c]
                        # make all the entries zero below (current_row,c)
                        for j in range(current_row + 1, height):
                            if mat[j, c] < 0:
                                mat[j, :] += (mat[current_row, :] * np.abs(mat[j, c]))
                            if mat[j, c] > 0:
                                mat[j, :] -= (mat[current_row, :] * mat[j, c])
                        # make all the entries zero above (current_row_c)
                        for j in range(current_row - 1, -1, -1):
                            if mat[j, c] < 0:
                                mat[j, :] += (mat[current_row, :] * np.abs(mat[j, c]))
                            if mat[j, c] > 0:
                                mat[j, :] -= (mat[current_row, :] * mat[j, c])
                        # we have found a pivot for this row, so now consider the next row
                        current_row += 1

    # returns the dimension of a simplicial complex
    def dimension(self, plex):
        max_dim = -1
        for simplex in plex:
            dim = len(simplex) - 1
            if dim > max_dim:
                max_dim = dim
        return max_dim

    # returns a list containing bases and a list containing dimensions for a simplicial complex
    def chain_groups(self, plex):
        Ck_dims = []  # Ck is the k-th chain group

        # get dimension of simplicial complex
        plex_dim = self.dimension(plex)

        # add empty lists to Ck_bases (these will hold the bases for the chain groups)
        # There are plex_dim+1 chain groups in the chain complex. We add 2 to this for the zero vector spaces at the beginning and end of the chain complex.
        Ck_bases = [[] for x in range(plex_dim + 3)]

        # populate nonzero chain groups, note: the first and last elements of Ck_bases will remain empty
        for k in range(len(Ck_bases)):
            for simplex in plex:
                if len(simplex) == k + 1:
                    Ck_bases[k + 1].append(simplex)

        # an example of Ck_bases:
        # [[], [[0], [1], [2], [3]], [[0, 1], [0, 2], [0, 3], [1, 2], [1, 3], [2, 3]], [[0, 1, 2], [0, 1, 3], [0, 2, 3], [1, 2, 3]], [[0, 1, 2, 3]], []]
        # [C_-1 = [], C_0, C_1, C_2, C_3, C_4 = []]

        # populate Ck_dims
        for chain_group_basis in Ck_bases:
            Ck_dims.append(len(chain_group_basis))

        # an example of Ck_dims:
        # [0, 4, 6, 4, 1, 0]
        # [dim(C_-1), dim(C_0), dim(C_1), dim(C_2), dim(C_3), dim(C_4)]

        return Ck_bases, Ck_dims

    # returns a list of boundary maps for a simplicial complex
    def boundary_maps(self, Ck_bases):
        boundary_maps = []
        for k in range(len(Ck_bases) - 1):
            # We have k in [0,1,2,3,4] and Ck_dims: [dim(C_-1), dim(C_0), dim(C_1), dim(C_2), dim(C_3), dim(C_4)]
            # We want to produce boundary_maps: [boundary_0 = 0, boundary_1, boundary_2, boundary_3, boundary_4 = 0]
            # So, boundary_k: C_k --> C_k-1 is a matrix of dimension dim(C_k-1) x dim(C_k).
            # A concrete example: boundary_0: C_0 --> C_-1 is of dimension dim(C_-1) x dim(C_0)
            mat = np.zeros((len(Ck_bases[k]), len(Ck_bases[k + 1])))

            # create zero maps at either end of the chain complex
            if k == 0 or k == len(Ck_bases) - 1:
                boundary_maps.append(mat)
            else:
                # build matrices according to the definition of the boundary map
                # for some k we want to produce boundary_k: Ck_bases[k+1] == C_k --> Ck_bases[k] == C_k-1
                # boundary_k is the matrix [bndry(x_0), bndry(x_1), ... , bndry(x_n)] where {x_0, x_1, ..., x_n} is a basis for C_k == Ck_bases[k+1]
                for simplex, simplex_index in zip(Ck_bases[k + 1], range(len(Ck_bases[k + 1]))):
                    entry = -1
                    for v_index in range(len(simplex)):
                        simplex_copy = list(simplex)
                        entry *= -1
                        simplex_copy.pop(v_index)
                        row = Ck_bases[k].index(simplex_copy)
                        mat[row, simplex_index] = entry
                boundary_maps.append(mat)

        return boundary_maps

    # returns a row reduced list of boundary maps given a list of boundary maps
    def rr_boundary_maps(self, boundary_maps):
        rr_boundary_maps = []
        for m in boundary_maps:
            m_copy = np.copy(m)
            self.row_reduce(m_copy)
            rr_boundary_maps.append(m_copy)
        return rr_boundary_maps

    # returns a list containing bases and a list containing dimensions for a simplicial complex
    def boundary_groups(self, rr_boundary_maps, boundary_maps):
        Bk_bases = []
        Bk_dims = []

        for k in range(len(rr_boundary_maps) - 1):  # this range makes boundary_maps[k+1] go from boundary_1 to boundary_4
            mat = rr_boundary_maps[k + 1]
            height = mat.shape[0]
            width = mat.shape[1]
            basis = []
            if mat.size == 0:  # if we get to the last element of boundary_maps which is the zero map and has no elements. The image of this map is the zero vector space.
                Bk_bases.append(basis)
                Bk_dims.append(len(basis))
            else:
                # get basis elements
                for row_index in range(height):
                    for col_index in range(width):
                        if mat[row_index, col_index] == 1:  # we have found a pivot value
                            basis.append(boundary_maps[k + 1][:,col_index].reshape((boundary_maps[k + 1][:,col_index].shape[0], 1)))  # element that we add to the basis is the column at col_index from the boundary matrix that hasn't been row reduced
                            break

                Bk_bases.append(basis)
                Bk_dims.append(len(basis))

        return Bk_bases, Bk_dims

    # returns a list with the indices for all the pivot columns: [(row_1, col_1), ..., (row_n, col_n)]
    def find_pivot_locations(self, row_reduced_matrix):
        height = row_reduced_matrix.shape[0]
        width = row_reduced_matrix.shape[1]
        pivot_col_locations = []
        for row_index in range(height):
            for col_index in range(width):
                if row_reduced_matrix[row_index, col_index] == 1:  # we have found a pivot value
                    pivot_col_locations.append((row_index, col_index))
                    break
        return pivot_col_locations

    # returns a list containing bases and a list containing dimensions for a simplicial complex
    def cycle_groups(self, rr_boundary_maps, Ck_bases):
        Zk_bases = []
        Zk_dims = []

        for k in range(len(rr_boundary_maps) - 1):  # this range makes boundary_maps[k] go from boundary_0 to boundary_3
            basis = []
            mat = rr_boundary_maps[k]
            width = mat.shape[1]
            if mat.size == 0:  # if we have a matrix that is the zero map, then Ker(boundary_k) = Ck = Ck_bases[k+1], so just use the existing basis (the standard basis in R^n) that we have for Ck as the basis for Zk
                for i in range(len(Ck_bases[k + 1])):
                    basis.append(np.zeros((len(Ck_bases[k + 1]), 1)))
                    basis[-1][i, 0] = 1
                Zk_bases.append(basis)
                Zk_dims.append(len(Ck_bases[k + 1]))
            else:
                # if we have a matrix that is not the zero map then we will find a basis by solving the equation boundary_K * x = 0 where x is a (n x 1) dimensional vector where n is the width of boundary_k
                # if we write the solution to this equation in parametric vector form, the vectors that span the solution set are a basis for the kernel. The code below computes these vectors.

                piv_locations = self.find_pivot_locations(mat)  # [(row_1, col_1), ..., (row_n, col_n)]
                non_piv_columns = set(range(width)) - set([piv[1] for piv in piv_locations])

                if non_piv_columns == set():  # if there is a pivot in every column, then the kernel is just the zero vector space, which has no basis
                    pass
                else:
                    for col in non_piv_columns:
                        basis.append(np.zeros((width, 1)))
                        # the next 3 lines contain the main logic that builds the set of vectors that is a basis for the kernel in the nontrivial case
                        basis[-1][col, 0] = 1
                        for piv in piv_locations:
                            basis[-1][piv[1], 0] = mat[piv[0], col] * -1

                Zk_bases.append(basis)
                Zk_dims.append(len(basis))

        return Zk_bases, Zk_dims

    def compute_homology(self, plex_type, dataset_name, dataset_type, num_points=-1):
        # get the simplicial complex and dataset to use
        if dataset_type == 'simple':
            plex = self.simplicial_complexes[dataset_name][plex_type]
        elif dataset_type == 'cloud':
            plex = self.simplicial_complexes[dataset_name][plex_type][num_points]

        # Ck is the k-th chain group
        # Ck_bases (for example): [C_-1 = [], C_0, C_1, C_2, C_3, C_4 = []]
        # Ck_dims (for example): [dim(C_-1), dim(C_0), dim(C_1), dim(C_2), dim(C_3), dim(C_4)]
        # this example will be continued throughout the comments in the code
        self.Ck_bases, self.Ck_dims = self.chain_groups(plex)

        # boundary_maps: [boundary_0 = 0, boundary_1, boundary_2, boundary_3, boundary_4 = 0]
        # each element in boundary_maps is a matrix
        self.boundary_matrices = self.boundary_maps(self.Ck_bases)

        # row reduce all of the matrices in boundary_maps
        self.rr_boundary_mats = self.rr_boundary_maps(self.boundary_matrices)

        # Bk is the k-th boundary group, Bk = Im(boundary_k+1)
        # Bk_bases: [basis(B_0), basis(B_1), basis(B_2), basis(B_3)]
        # Bk_dims: [dim(B_0), dim(B_1), dim(B_2), dim(B_3)]
        # note: we find basis(B_k) and dim(B_k) only for k where C_k is not the zero vector space
        self.Bk_bases, self.Bk_dims = self.boundary_groups(self.rr_boundary_mats, self.boundary_matrices)

        # Zk is the k-th cycle group, Zk = Ker(boundary_k)
        # Zk_bases: [basis(Z_0), basis(Z_1), basis(Z_2), basis(Z_3)]
        # Zk_dims: [dim(Z_0), dim(Z_1), dim(Z_2), dim(Z_3)]
        # note: we find basis(Z_k) and dim(Z_k) only for k where C_k is not the zero vector space
        self.Zk_bases, self.Zk_dims = self.cycle_groups(self.rr_boundary_mats, self.Ck_bases)

        self.Hk_bases, self.Hk_dims = self.homology_groups(self.Bk_dims, self.Zk_dims, self.Bk_bases, self.Zk_bases)

    def homology_groups(self, Bk_dims, Zk_dims, Bk_bases, Zk_bases):
        Hk_dims = []
        Hk_bases = []

        for Bk_dim, Zk_dim in zip(Bk_dims, Zk_dims):  # compute betti numbers
            Hk_dim = Zk_dim - Bk_dim
            Hk_dims.append(Hk_dim)

        for Bk_basis, Zk_basis in zip(Bk_bases, Zk_bases):
            if len(Zk_basis) == 0:  # if Zk is the zero vector space then Hk is as well
                Hk_bases.append([])
            elif len(Bk_basis) == 0:  # if we mod out by the zero vector space then Hk is all of Zk
                Hk_bases.append(Zk_basis)
            else:
                # this block of code computess a basis for Hk in non-trivial cases, see thesis for a description
                new_basis_for_ker = Bk_basis + Zk_basis
                utility_matrix = np.hstack(tuple(Bk_basis + Zk_basis))

                self.row_reduce(utility_matrix)

                piv_locs = self.find_pivot_locations(utility_matrix)
                non_piv_columns = list(set(range(utility_matrix.shape[1])) - set([piv[1] for piv in piv_locs]))

                # delete the columns that don't have pivots from the basis
                for index in sorted(non_piv_columns, reverse=True):
                    del new_basis_for_ker[index]

                # delete the basis vectors of the boundary group from the new basis
                for index in sorted(list(range(len(Bk_basis))), reverse=True):
                    del new_basis_for_ker[index]

                Hk_bases.append(new_basis_for_ker)

        return Hk_bases, Hk_dims

    def scatter_plot_2d(self, dataset_name, num_points):
        if dataset_name == 'figure_8':
            x_range = [-2, 2]
        else:
            x_range = [-1.5, 1.5]
        y_range = [-1.5, 1.5]
        trace = Scatter(
            x=self.datasets[dataset_name][num_points][:, 0].tolist(),
            y=self.datasets[dataset_name][num_points][:, 1].tolist(),
            mode='markers',
            name='point_cloud'
        )
        data = [trace]
        layout = Layout(title=dataset_name + ' ' + str(num_points),
                        xaxis=dict(range=x_range),
                        yaxis=dict(range=y_range)
                        )
        fig = dict(data=data, layout=layout)
        pt.offline.plot(fig, filename=dataset_name + '_' + str(num_points) + '.html')

    def scatter_plot_3d(self, dataset_name, num_points):
        trace = Scatter3d(
            x=self.datasets[dataset_name][num_points][:, 0].tolist(),
            y=self.datasets[dataset_name][num_points][:, 1].tolist(),
            z=self.datasets[dataset_name][num_points][:, 2].tolist(),
            mode='markers',
            name='point_cloud'
        )
        data = [trace]
        layout = Layout(title=dataset_name + ' ' + str(num_points)  # ,
                        # xaxis=dict(range=[-1.5, 1.5]),
                        # yaxis=dict(range=[-1.5, 1.5]),
                        # zaxis=dict(range=[-1.5, 1.5])
                        )
        fig = dict(data=data, layout=layout)
        pt.offline.plot(fig, filename=dataset_name + '_' + str(num_points) + '.html')

if __name__ == '__main__':
    h = Homology()
    h.main()