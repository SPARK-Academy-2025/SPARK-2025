import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.special import legendre
from matplotlib import cm
from .utils.utils import *
from .cldice import soft_dice_cldice, soft_skel

def get_background_index_(x, no_idx_select=10, label=0):
    all_coordinate = torch.nonzero(x == label, as_tuple=False)
    random_index = []
    all_coordinate_idx = all_coordinate.shape[0]
    selected_coordinate_idx = np.random.randint(0, all_coordinate_idx, no_idx_select)
    center_coor_list = [all_coordinate[x] for x in selected_coordinate_idx]
    return center_coor_list

def generate_multi_gauss_mask(x, center_coor_list, var):
    dim = len(x.shape) - 2
    if dim ==3:
        xx, yy, zz = np.meshgrid(np.arange(x.shape[2]), np.arange(x.shape[3]), np.arange(x.shape[4]), indexing='ij')
    elif dim ==2:
        xx, yy,  = np.meshgrid(np.arange(x.shape[2]), np.arange(x.shape[3]), indexing='ij')
    multi_gaussian_map = torch.zeros_like(x)
    binary_mask_for_weighted_loss = torch.zeros_like(x)
    for i, coor in enumerate(center_coor_list):
        if dim == 3:
            coor_x, coor_y, coor_z = [int(coor[2]), int(coor[3]), int(coor[4])]
            distances_squared = ((xx - coor_x) ** 2 / (2 * float(np.random.normal(loc=var, scale=0.5, size=1)) ** 2) +
                                 (yy - coor_y) ** 2 / (2 * float(np.random.normal(loc=var, scale=0.5, size=1)) ** 2) +
                                 (zz - coor_z) ** 2 / (2 * float(np.random.normal(loc=var, scale=0.5, size=1)) ** 2))
        elif dim == 2:
            coor_x, coor_y = [int(coor[2]), int(coor[3])]
            distances_squared = ((xx - coor_x) ** 2 / (2 * float(np.random.normal(loc=var, scale=0.5, size=1)) ** 2) +
                                 (yy - coor_y) ** 2 / (2 * float(np.random.normal(loc=var, scale=0.5, size=1)) ** 2) )

        gaussian_map = np.exp(-distances_squared).astype(np.float32)

        # Normalize the values to the range [0, 1]
        gaussian_map = (gaussian_map - np.min(gaussian_map)) / (np.max(gaussian_map) - np.min(gaussian_map))

        mask_1 = torch.from_numpy(gaussian_map[np.newaxis, np.newaxis, ...]).to(x.device)
        mask_1 = mask_1 * np.random.randint(10)/10

        multi_gaussian_map += mask_1

    # multi_gaussian_map = (multi_gaussian_map - multi_gaussian_map.min()) / (multi_gaussian_map.max() - multi_gaussian_map.min() + 0.000001)
    binary_mask_for_weighted_loss[multi_gaussian_map>=0.05] = 1

    return multi_gaussian_map, binary_mask_for_weighted_loss

def random_mask(x, map_size, inpaint_type, poly, order, isCL=True, var=5, idx=None):
    dim = int(len(x.shape) - 2)
    if isCL:
        x_skel = soft_skel(x, iter_=3)
    else:
        x_skel = x
    x_np = x_skel.cpu().detach().numpy()
    foreground_coords = np.column_stack(np.where(x_np != 0))
    if len(foreground_coords) == 0:
        # print(ValueError("Label map does not have any foreground."))
        return torch.zeros_like(x), torch.zeros_like(x), torch.zeros_like(x)
    if idx is not None:
        rand_index = idx * len(foreground_coords)//11
    else:
        rand_index = np.random.randint(0, len(foreground_coords))
    if dim ==3:
        center_x, center_y, center_z = foreground_coords[rand_index][2:5]
    elif dim ==2:
        center_x, center_y = foreground_coords[rand_index][2:4]
    delta = map_size
    start_x = center_x - delta // 2
    start_y = center_y - delta // 2
    start_x = max(0, min(start_x, x_np.shape[2] - delta))
    start_y = max(0, min(start_y, x_np.shape[3] - delta))
    if dim == 3:
        start_z = center_z - delta // 2
        start_z = max(0, min(start_z, x_np.shape[4] - delta))
        slicing_foreground = np.s_[:, :, start_x:start_x+delta, start_y:start_y+delta, start_z:start_z+delta]
    elif dim == 2:
        slicing_foreground = np.s_[:, :, start_x:start_x + delta, start_y:start_y + delta]

    if inpaint_type in ['polynomial_bi', 'polynomial_bi_binary']:
        # get the coordinate of background first, in order to mask on background later
        x_background = 1 - x
        background_coordinate = torch.nonzero(x_background == 1, as_tuple=False)
        random_index = background_coordinate[np.random.randint(0, background_coordinate.shape[0])]
        if dim == 3:
            center_x_1, center_y_1, center_z_1 = [random_index[2], random_index[3], random_index[4],]
        elif dim == 2:
            center_x_1, center_y_1 = [random_index[2], random_index[3], ]
        center_x_1 = int(center_x_1)
        center_y_1 = int(center_y_1)
        start_x_1 =  center_x_1 - delta//2
        start_y_1 =  center_y_1- delta//2
        start_x_1 = max(0, min(start_x_1, x_np.shape[2] - delta))
        start_y_1 = max(0, min(start_y_1, x_np.shape[3] - delta))
        if dim == 3:
            center_z_1 = int(center_z_1)
            start_z_1 = center_z_1 - delta//2
            start_z_1 = max(0, min(start_z_1, x_np.shape[4] - delta))
            slicing_background = np.s_[:, :, start_x_1:start_x_1 + delta, start_y_1:start_y_1 + delta,
                                 start_z_1:start_z_1 + delta]
        elif dim == 2:
            slicing_background = np.s_[:, :, start_x_1:start_x_1 + delta, start_y_1:start_y_1 + delta]


    '''remove poly on foreground, marked with 'polynomial_xxx' '''
    map, coeffs = poly.get_mask(order=order)
    map = 1.5 * map - 0.5
    map[map<0] =0
    mask = torch.zeros_like(x)
    mask[slicing_foreground] = 1
    x_masked = x.clone().detach()

    map_tensor = torch.from_numpy(map[np.newaxis, np.newaxis, ...]).to(x_masked.device).to(x_masked.dtype)
    
    assert x_masked[slicing_foreground].shape == map_tensor.shape, \
        f"Shape mismatch: x_masked slice {x_masked[slicing_foreground].shape}, map {map_tensor.shape}"

    # Then apply the mask
    x_masked[slicing_foreground] *= map_tensor

    '''add poly in background, marked with '_bi' '''
    if inpaint_type in ['polynomial_bi', 'polynomial_bi_binary']:
        map_2, coeffs_2 = poly.get_mask(order=order, p_low=0.05, p_high=0.5)
        map_2 = 1.5 * map_2 - 0.5
        map_2[map_2 < 0] = 0
        mask[slicing_background] = 0.1
        x_masked[slicing_background] += \
            (1 - x_masked[slicing_background]) * torch.from_numpy(map_2[np.newaxis, np.newaxis, ...]).to(x_masked.device)


    '''add gauss dots in background, marked with '_gauss' '''
    if inpaint_type in ['polynomial_gauss', 'polynomial_gauss_binary',]:
        no_idx_select= np.random.randint(1, 10)
        center_coor_list = get_background_index_(x, no_idx_select=no_idx_select, label=0)
        multi_gaussian_map, binary_mask_for_weighted_loss = generate_multi_gauss_mask(x, center_coor_list, var=3)

        x_masked = x_masked + multi_gaussian_map
        x_masked[x_masked>1] = 1
        mask[binary_mask_for_weighted_loss == 1] = 1

    '''remove square boxes on foreground, marked with '_binary' '''
    if inpaint_type in ['polynomial_binary', 'polynomial_gauss_binary', 'polynomial_bi_binary']:
        if isCL:
            x_skel = soft_skel(x, iter_=3)
        else:
            x_skel = x
        x_np = x_skel.cpu().detach().numpy()
        foreground_coords = np.column_stack(np.where(x_np != 0))
        # if len(foreground_coords) == 0:
        #     raise ValueError("Label map does not have any foreground.")
        if idx is not None:
            rand_index = idx * len(foreground_coords)//15
        else:
            rand_index = np.random.randint(0, len(foreground_coords))

        if dim == 3:
            center_x_2, center_y_2, center_z_2 = foreground_coords[rand_index][2:5]
        elif dim == 2:
            center_x_2, center_y_2 = foreground_coords[rand_index][2:4]
        delta_1 = 30
        start_x_2 = center_x_2 - delta_1 //2
        start_y_2 = center_y_2 - delta_1 // 2
        start_x_2 = max(0, min(start_x_2, x_np.shape[2] - delta_1))
        start_y_2 = max(0, min(start_y_2, x_np.shape[3] - delta_1))
        if dim == 3:
            start_z_2 = center_z_2 - delta_1 // 2
            start_z_2 = max(0, min(start_z_2, x_np.shape[4] - delta_1))
            slicing_foreground_1 = np.s_[:,:, start_x_2:start_x_2+delta_1, start_y_2:start_y_2+delta_1, start_z_2:start_z_2+delta_1]
        elif dim == 2:
            slicing_foreground_1 = np.s_[:,:, start_x_2:start_x_2+delta_1, start_y_2:start_y_2+delta_1]

        x_masked[slicing_foreground_1] = 0
        mask[slicing_foreground_1] = 1

    return x_masked, mask, x_skel

class Polynomial:
    def __init__(self, map_size, order=10, dim=2, basis_type='legendre'):
        """
        Initialize the polynomial with its type and coefficients for each dimension.
        :param basis_type: String, the type of polynomial ("hermite", "legendre", etc.)
        :param coefficients: List of lists, coefficients for the polynomial in each dimension.
        """
        self.basis_type = basis_type
        self.dimensions = dim
        self.order = order
        self.map_size = map_size

        self.basis = {}
        if dim ==2:
            self.init_2d_basis(self.order)
        elif dim ==3:
            self.init_3d_basis(self.order)
        else:
            assert 'WRONG INPUT DIMENSION'

    def legendre_polynomial(self, n, x):
        if n == 0:
            return np.ones_like(x)
        elif n == 1:
            return x
        else:
            return ((2 * n - 1) * x * self.legendre_polynomial(n - 1, x) - (n - 1) * self.legendre_polynomial(n - 2, x)) / n

    def chebyshev_first_kind_iter(self, n, x):
        x = np.array(x, dtype=np.float64)
        if n == 0:
            return np.ones_like(x)
        elif n == 1:
            return x
        else:
            T0 = np.ones_like(x)
            T1 = x
            for _ in range(2, n+1):
                T2 = 2 * x * T1 - T0
                T0, T1 = T1, T2
            return T1

    def hermite_polynomial_physicist(self, n, x):
        if n == 0:
            return np.ones_like(x)
        elif n == 1:
            return 2 * x
        else:
            return 2 * x * self.hermite_polynomial_physicist(n - 1, x) - 2 * (n - 1) * self.hermite_polynomial_physicist(n - 2, x)


    def hermite_functions(self,n, x):
        return (2**n * np.math.factorial(n) * np.sqrt(np.pi))**-0.5 * self.hermite_polynomial_physicist(n, x) * np.exp(-x**2 / 2)

    def hermite_functions_nomalized(self,n, x):
        """ scale the x axis to [-1, 1] , the scale should be the 95 precentile of the nth hermite function"""
        scale = 2.5 * np.sqrt(n+1)
        x = x * scale
        return self.hermite_functions(n, x)

    def init_2d_basis(self, order):
        x = np.linspace(-1, 1, self.map_size)
        y = np.linspace(-1, 1, self.map_size)
        X, Y = np.meshgrid(x, y)

        choice = np.eye(order, dtype=int).tolist()
        for i in range(order):
            for j in range(order):
                # Create a polynomial with random coefficients
                basis = self.evaluate([X, Y], [choice[i], choice[j]])
                key_name = str(i) + '-' + str(j)
                self.basis.update({
                    key_name: basis
                })

    def init_3d_basis(self, order):
        x = np.linspace(-1, 1, self.map_size)
        y = np.linspace(-1, 1, self.map_size)
        z = np.linspace(-1, 1, self.map_size)
        X, Y, Z = np.meshgrid(x, y, z)

        choice = np.eye(order, dtype=int).tolist()
        for i in range(order):
            for j in range(order):
                for k in range(order):
                    # Create a polynomial with random coefficients
                    basis = self.evaluate(points=[X, Y, Z], coefficients=[choice[i], choice[j], choice[k]])
                    key_name = str(i) + '-' + str(j) + '-' + str(k)
                    self.basis.update({
                        key_name: basis
                    })

    def evaluate_basis(self, n, x):
        """
        Evaluate the n-th polynomial of the given basis type at points x.
        :param n: Integer, the order of the polynomial.
        :param x: Array of points at which to evaluate the polynomial.
        :return: Evaluated polynomial at points x.
        """
        if self.basis_type == "hermite":
            return self.hermite_functions_nomalized(n, x)
        elif self.basis_type == "legendre":
            return self.legendre_polynomial(n, x)
        elif self.basis_type == 'chebyshev':
            return self.chebyshev_first_kind_iter(n, x)
        else:
            raise ValueError("Unsupported basis type")

    def evaluate(self, points, coefficients):
        """
        Evaluate the polynomial at given points in space.
        :param points: Array-like, the points in space at which to evaluate the polynomial.
                       Should have the same number of dimensions as there are sets of coefficients.
        :return: The polynomial evaluated at the given points.
        """
        if len(points) != self.dimensions:
            raise ValueError("Points dimensionality does not match the polynomial dimensions")

        result = np.ones_like(points[0])
        for dim in range(self.dimensions):
            dim_result = np.zeros_like(points[dim])
            for order, coeff in enumerate(coefficients[dim]):
                dim_result += coeff * self.evaluate_basis(order, points[dim])
            result *= dim_result

        return result

    def get_mask(self, order, coeffs_type='gauss', p_low=0.05, p_high=0.95):
        FLAG= True
        while FLAG:
            if coeffs_type == 'gauss':
                if self.dimensions ==2:
                    coeffs = np.random.randn(order, order)
                    r = np.zeros((self.map_size, self.map_size))

                    for i in range(order):
                        for j in range(order):
                            key_name = str(i) + '-' + str(j)
                            r += coeffs[i, j] * self.basis[key_name]

                elif self.dimensions ==3:
                    coeffs = np.random.randn(order, order, order)
                    r = np.zeros((self.map_size, self.map_size, self.map_size))

                    for i in range(order):
                        for j in range(order):
                            for k in range(order):
                                key_name = str(i) + '-' + str(j) + '-' + str(k)
                                r += coeffs[i, j, k] * self.basis[key_name]

            r = norm_ten(r)
            p = norm_ten(r, binary=True).sum() / (self.map_size ** self.dimensions)
            if p <= p_high and p >= p_low:
                FLAG = False

        return r, coeffs


if __name__ == '__main__':
    order_all = 10
    poly = Polynomial(map_size=64, order=order_all)

    for i in range(10):
        mask, coeff = poly.get_mask(order=4)
        mask = norm_ten(mask)
        plot_3d_np(mask[np.newaxis, ...])