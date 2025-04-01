import HspyPrep
import numpy as np
import matplotlib.pyplot as plt
from skimage import io, color
from skimage.metrics import structural_similarity as ssim
import matplotlib.patches as patches
import pickle
from lmfit.models import GaussianModel, ConstantModel, LorentzianModel, VoigtModel
from matplotlib.gridspec import GridSpec
import traceback
import seaborn as sns
from scipy.signal import find_peaks, peak_widths
import os
import pandas as pd
from skimage.exposure import match_histograms

class CondAns:
    def __init__(self, data_dict: dict, ref, load_mapping=False):
        self.best_model = None
        self.params_fit = None
        self.best_fit = None
        self.data_dict = data_dict
        self.ref = ref
        if load_mapping:
            with open("data_coordinates.pkl", "rb") as file:
                self.data_coordinates = pickle.load(file)

    def map_all_pixels(self, window_size, max_disp, ref):
        image_ref = self.data_dict[ref].get_live_scan()
        mapping_save = dict()
        mapping = ''
        for key, value in self.data_dict.items():
            value.live_scan = match_histograms(value.live_scan, image_ref)
        for key in self.data_dict.keys():
            if key == ref:
                mapping_save['ref'] = key
                continue
            mapping = CondAns.map_pixels(image_ref, self.data_dict[key].get_live_scan(), window_size=window_size,
                                         search_radius=max_disp)
            mapping_save[key] = mapping
        print(mapping_save)
        mapping_save['ref'] = list(mapping.keys())
        with open("data_coordinates.pkl", "wb") as file:
            pickle.dump(mapping_save, file)
            self.data_coordinates = mapping_save

    def plot_all_pixels(self, figsize=(20, 10), save=False, filename=None):
        list_of_exp = list(self.data_dict.keys())

        for key_coord in self.data_coordinates['ref']:
            colors = sns.color_palette("inferno", len(list_of_exp))
            n_images = len(list_of_exp)
            fig = plt.figure(figsize=figsize)
            gs = GridSpec(6, 7, figure=fig, wspace=0.1, hspace=0.2)
            ax_main = fig.add_subplot(gs[:, 0:4])
            image_axes = dict()
            n_cols = 3

            for i, (key, value) in enumerate(self.data_dict.items(), start=0):
                row = (i // n_cols) * 2
                col = 4 + (i % n_cols)
                ax_im = fig.add_subplot(gs[row:row + 2, col])
                image_axes[key] = ax_im

            coord_x = []
            coord_y = []
            for i in self.data_coordinates.keys():
                if i == 'ref':
                    coord_x.append(key_coord[0])
                    coord_y.append(key_coord[1])
                    CondAns.__plot_image_with_rect(image_axes[self.ref], self.data_dict[self.ref].get_live_scan(),
                                                   (coord_x[-1], coord_y[-1]), self.ref)
                else:
                    temp_coord = self.data_coordinates[i].get(key_coord)
                    if temp_coord is None:
                        temp_coord = (0, 0)
                    coord_x.append(temp_coord[0])
                    coord_y.append(temp_coord[1])
                    CondAns.__plot_image_with_rect(image_axes[i], self.data_dict[i].get_live_scan(),
                                                   (coord_x[-1], coord_y[-1]), i)
            for idx, (temp, color, x, y) in enumerate(zip(list(self.data_dict.keys()), colors, coord_y, coord_x)):
                if x == 0 and y == 0:
                    continue
                wavelengths = self.data_dict[temp].get_wavelengths()[::-1]
                intensity = self.data_dict[temp].get_numpy_spectra()[x][y][::-1]
                ax_main.plot(wavelengths, intensity, color=color, linewidth=2, label=f'{temp} nA')

            ax_main.set_xlabel('Wavelength (nm)', fontsize=18, labelpad=15)
            ax_main.set_ylabel('Intensity (a.u.)', fontsize=18, labelpad=15)

            ax_main.tick_params(axis='both', which='major', labelsize=14, length=6, width=1.5)
            ax_main.tick_params(axis='both', which='minor', length=4, width=1)

            # Add top x-axis for energy scale
            secax = ax_main.secondary_xaxis('top')
            secax.set_xlabel('Energy (eV)', fontsize=18, labelpad=15)
            secax.tick_params(axis='x', labelsize=14, length=6, width=1.5)

            wavelength_ticks = np.linspace(min(wavelengths), max(wavelengths), num=6)
            energy_ticks = np.linspace(wavelengths[0], wavelengths[-1], num=5)
            secax.set_xticks(energy_ticks)
            secax.set_xticklabels([f'{1239.84193 / wl:.2f}' for wl in energy_ticks])

            ax_main.grid(True)
            ax_main.grid(True, which='both', color='black', linestyle='--', linewidth=0.3)
            for spine in ax_main.spines.values():
                spine.set_edgecolor('black')
                spine.set_linewidth(1.5)
            ax_main.axvspan(950, 1000, color='green', alpha=0.15, label="950-1000 nm")
            ax_main.axvspan(870, 940, color='blue', alpha=0.15, label="870-940 nm")

            ax_main.legend()
            if save:
                folder = f'./{filename}'
                os.makedirs(folder, exist_ok=True)
                plt.savefig(f'{folder}/{key_coord[0]}_{key_coord[1]}.png', dpi=400)
                plt.show()
            plt.close()

    def plot_all_pixels_with_fitting(self, figsize=(20, 10), save=False, filename=None, fit_func=VoigtModel,
                                     peaks='Automatic'):
        '''

        :param figsize:
        :param save:
        :param filename:
        :param fit_func: define a fit function which the data can be fitted with thats
        :param peaks: The number of peaks will be defined automatically; however, you can define estimations about the number of peaks that you may have.
        :return:
        '''
        list_of_exp = list(self.data_dict.keys())
        for key_coord in self.data_coordinates['ref']:
            colors = sns.color_palette("inferno", len(list_of_exp))
            n_images = len(list_of_exp)
            fig = plt.figure(figsize=figsize)
            gs = GridSpec(6, 7, figure=fig, wspace=0.1, hspace=0.2)
            ax_main = fig.add_subplot(gs[:, 0:4])
            image_axes = dict()
            n_cols = 3

            for i, (key, value) in enumerate(self.data_dict.items(), start=0):
                row = (i // n_cols) * 2
                col = 4 + (i % n_cols)
                ax_im = fig.add_subplot(gs[row:row + 2, col])
                image_axes[key] = ax_im

            coord_x = []
            coord_y = []
            for i in self.data_coordinates.keys():
                if i == 'ref':
                    coord_x.append(key_coord[0])
                    coord_y.append(key_coord[1])
                    CondAns.__plot_image_with_rect(image_axes[self.ref], self.data_dict[self.ref].get_live_scan(),
                                                   (coord_x[-1], coord_y[-1]), self.ref)
                else:
                    temp_coord = self.data_coordinates[i].get(key_coord)
                    if temp_coord is None:
                        temp_coord = (0, 0)
                    coord_x.append(temp_coord[0])
                    coord_y.append(temp_coord[1])
                    CondAns.__plot_image_with_rect(image_axes[i], self.data_dict[i].get_live_scan(),
                                                   (coord_x[-1], coord_y[-1]), i)
            for idx, (temp, color, x, y) in enumerate(zip(list(self.data_dict.keys()), colors, coord_y, coord_x)):
                if x == 0 and y == 0:
                    continue
                wavelengths = self.data_dict[temp].get_wavelengths()[::-1]
                intensity = self.data_dict[temp].get_numpy_spectra()[x][y][::-1]
                # ax_main.plot(wavelengths, intensity, color=color, linewidth=2, label=f'{temp} nA')
                if peaks == 'Automatic':
                    peaks_indices, properties = find_peaks(intensity, height=80, prominence=5, distance=20)
                    peak_positions = wavelengths[peaks_indices]
                    peak_heights = intensity[peaks_indices]

                    sorted_idx = np.argsort(peak_heights)[::-1]
                    peak_positions = peak_positions[sorted_idx]
                    peak_heights = peak_heights[sorted_idx]

                    r2_list = []
                    params_list = []
                    models = []
                    result = ''

                    model = ConstantModel(prefix='bkg_')
                    params = model.make_params(bkg_c=0)
                    for m, (pos, height) in enumerate(zip(peak_positions, peak_heights)):
                        prefix = f'g{m}_'
                        gauss = fit_func(prefix=prefix)
                        model += gauss
                        params.update(gauss.make_params())
                        params[f'{prefix}amplitude'].set(value=height, min=0)
                        params[f'{prefix}center'].set(value=pos)
                        params[f'{prefix}sigma'].set(value=1, min=0.1)

                        result = model.fit(intensity, params, x=wavelengths)

                        ss_total = np.sum((intensity - np.mean(intensity)) ** 2)
                        ss_residual = np.sum(result.residual ** 2)
                        r_squared = 1 - (ss_residual / ss_total)
                        r2_list.append(r_squared)
                        params_list.append(result.params)
                        models.append(model)
                        print(f"Using {m + 1} peak(s): R² = {r_squared:.4f}")

                        if len(peak_positions) == 1:
                            self.best_fit = m
                            self.params_fit = params_list[-1]
                            self.best_model = models[-1]
                            break

                        if len(r2_list) == len(peak_positions):
                            if r2_list[-1] - r2_list[-2] < 0.01:
                                self.best_fit = m
                                self.params_fit = params_list[-2]
                                self.best_model = models[-2]
                            else:
                                self.best_fit = m + 1
                                self.params_fit = params_list[-1]
                                self.best_model = models[-1]

                        if len(r2_list) > 1 and r2_list[-1] - r2_list[-2] < 0.01 and r2_list[-2] > 0.9:
                            self.best_fit = m
                            self.params_fit = params_list[-2]
                            self.best_model = models[-2]
                            break

                    try:
                        result = self.best_model.fit(intensity, params, x=wavelengths)
                        ss_total = np.sum((intensity - np.mean(intensity)) ** 2)
                        ss_residual = np.sum(result.residual ** 2)
                        r_squared = 1 - (ss_residual / ss_total)
                        ax_main.plot(wavelengths, result.best_fit, color=color, linewidth=2,
                                     label=f'{temp} nA - R²={r_squared:.4f}')
                    except Exception as e:
                        print("An error occurred during fitting:")
                        traceback.print_exc()

            ax_main.set_xlabel('Wavelength (nm)', fontsize=18, labelpad=15)
            ax_main.set_ylabel('Intensity (a.u.)', fontsize=18, labelpad=15)

            ax_main.tick_params(axis='both', which='major', labelsize=14, length=6, width=1.5)
            ax_main.tick_params(axis='both', which='minor', length=4, width=1)

            secax = ax_main.secondary_xaxis('top')
            secax.set_xlabel('Energy (eV)', fontsize=18, labelpad=15)
            secax.tick_params(axis='x', labelsize=14, length=6, width=1.5)

            wavelength_ticks = np.linspace(min(wavelengths), max(wavelengths), num=6)
            energy_ticks = np.linspace(wavelengths[0], wavelengths[-1], num=5)
            secax.set_xticks(energy_ticks)
            secax.set_xticklabels([f'{1239.84193 / wl:.2f}' for wl in energy_ticks])

            ax_main.grid(True)
            ax_main.grid(True, which='both', color='black', linestyle='--', linewidth=0.3)
            for spine in ax_main.spines.values():
                spine.set_edgecolor('black')
                spine.set_linewidth(1.5)
            ax_main.axvspan(950, 1000, color='green', alpha=0.15, label="950-1000 nm")
            ax_main.axvspan(870, 940, color='blue', alpha=0.15, label="870-940 nm")

            ax_main.legend()
            if save:
                folder = f'./{filename}'
                os.makedirs(folder, exist_ok=True)
                plt.savefig(f'{folder}/{key_coord[0]}_{key_coord[1]}_fitted.png', dpi=400)
                plt.show()
            plt.close()

    def single_exp_run_plot(self, exp_key, figsize=(20, 10), save=False, filename=None, fit_func=VoigtModel,
                            peaks='Automatic'):
        exp = self.data_dict.get(exp_key)
        mapping = dict()
        for i in range(exp.get_live_scan().shape[0]):
            for j in range(exp.get_live_scan().shape[1]):
                mapping[(i, j)] = (i, j)

        for key_coord in mapping:
            fig = plt.figure(figsize=figsize)
            gs = GridSpec(6, 7, figure=fig, wspace=0.1, hspace=0.2)
            ax_main = fig.add_subplot(gs[:, :])
            x = key_coord[1]
            y = key_coord[0]

            wavelengths = self.data_dict[exp_key].get_wavelengths()[::-1]
            intensity = self.data_dict[exp_key].get_numpy_spectra()[x][y][::-1]
            ax_main.plot(wavelengths, intensity, color='red', linewidth=2)
            if peaks == 'Automatic':
                peaks_indices, properties = find_peaks(intensity, height=80, prominence=5, distance=20)
                peak_positions = wavelengths[peaks_indices]
                peak_heights = intensity[peaks_indices]

                sorted_idx = np.argsort(peak_heights)[::-1]
                peak_positions = peak_positions[sorted_idx]
                peak_heights = peak_heights[sorted_idx]

                r2_list = []
                params_list = []
                models = []
                result = ''

                model = ConstantModel(prefix='bkg_')
                params = model.make_params(bkg_c=0)
                for m, (pos, height) in enumerate(zip(peak_positions, peak_heights)):
                    prefix = f'g{m}_'
                    gauss = fit_func(prefix=prefix)
                    model += gauss
                    params.update(gauss.make_params())
                    params[f'{prefix}amplitude'].set(value=height, min=0)
                    params[f'{prefix}center'].set(value=pos)
                    params[f'{prefix}sigma'].set(value=1, min=0.1)

                    result = model.fit(intensity, params, x=wavelengths)

                    ss_total = np.sum((intensity - np.mean(intensity)) ** 2)
                    ss_residual = np.sum(result.residual ** 2)
                    r_squared = 1 - (ss_residual / ss_total)
                    r2_list.append(r_squared)
                    params_list.append(result.params)
                    models.append(model)
                    print(f"Using {m + 1} peak(s): R² = {r_squared:.4f}")

                    if len(peak_positions) == 1:
                        self.best_fit = m
                        self.params_fit = params_list[-1]
                        self.best_model = models[-1]
                        break

                    if len(r2_list) == len(peak_positions):
                        if r2_list[-1] - r2_list[-2] < 0.01:
                            self.best_fit = m
                            self.params_fit = params_list[-2]
                            self.best_model = models[-2]
                        else:
                            self.best_fit = m + 1
                            self.params_fit = params_list[-1]
                            self.best_model = models[-1]

                    if len(r2_list) > 1 and r2_list[-1] - r2_list[-2] < 0.01 and r2_list[-2] > 0.9:
                        self.best_fit = m
                        self.params_fit = params_list[-2]
                        self.best_model = models[-2]
                        break

                try:
                    result = self.best_model.fit(intensity, params, x=wavelengths)

                    ss_total = np.sum((intensity - np.mean(intensity)) ** 2)
                    ss_residual = np.sum(result.residual ** 2)
                    r_squared = 1 - (ss_residual / ss_total)

                    ax_main.plot(wavelengths, result.best_fit, color='blue', linewidth=2,
                                 label=f'R²={r_squared:.4f}')
                except Exception as e:
                    print("An error occurred during fitting:")
                    traceback.print_exc()

            ax_main.set_xlabel('Wavelength (nm)', fontsize=18, labelpad=15)
            ax_main.set_ylabel('Intensity (a.u.)', fontsize=18, labelpad=15)

            ax_main.tick_params(axis='both', which='major', labelsize=14, length=6, width=1.5)
            ax_main.tick_params(axis='both', which='minor', length=4, width=1)

            secax = ax_main.secondary_xaxis('top')
            secax.set_xlabel('Energy (eV)', fontsize=18, labelpad=15)
            secax.tick_params(axis='x', labelsize=14, length=6, width=1.5)

            wavelength_ticks = np.linspace(min(wavelengths), max(wavelengths), num=6)
            energy_ticks = np.linspace(wavelengths[0], wavelengths[-1], num=5)
            secax.set_xticks(energy_ticks)
            secax.set_xticklabels([f'{1239.84193 / wl:.2f}' for wl in energy_ticks])

            ax_main.grid(True)
            ax_main.grid(True, which='both', color='black', linestyle='--', linewidth=0.3)
            for spine in ax_main.spines.values():
                spine.set_edgecolor('black')
                spine.set_linewidth(1.5)
            ax_main.axvspan(950, 1000, color='green', alpha=0.15, label="950-1000 nm")
            ax_main.axvspan(870, 940, color='blue', alpha=0.15, label="870-940 nm")

            ax_main.legend()
            if save:
                folder = f'./{filename}'
                os.makedirs(folder, exist_ok=True)
                plt.savefig(f'{folder}/{key_coord[0]}_{key_coord[1]}_fitted.png', dpi=400)
                plt.show()

            plt.close()
        pass

    def single_exp_run_fitting(self, exp_key, save=False, filename=None, fit_func=VoigtModel,
                               peaks='Automatic'):
        exp = self.data_dict.get(exp_key)
        params_data = []
        mapping = dict()
        for i in range(exp.get_live_scan().shape[0]):
            for j in range(exp.get_live_scan().shape[1]):
                mapping[(i, j)] = (i, j)

        for key_coord in mapping:
            x = key_coord[1]
            y = key_coord[0]
            print(f'processing {x}_{y}')

            wavelengths = self.data_dict[exp_key].get_wavelengths()[::-1]
            intensity = self.data_dict[exp_key].get_numpy_spectra()[x][y][::-1]
            if peaks == 'Automatic':
                peaks_indices, properties = find_peaks(intensity, height=80, prominence=5, distance=20)
                peak_positions = wavelengths[peaks_indices]
                peak_heights = intensity[peaks_indices]

                sorted_idx = np.argsort(peak_heights)[::-1]
                peak_positions = peak_positions[sorted_idx]
                peak_heights = peak_heights[sorted_idx]

                r2_list = []
                params_list = []
                models = []
                result = ''

                model = ConstantModel(prefix='bkg_')
                params = model.make_params(bkg_c=0)
                for m, (pos, height) in enumerate(zip(peak_positions, peak_heights)):
                    prefix = f'g{m}_'
                    gauss = fit_func(prefix=prefix)
                    model += gauss
                    params.update(gauss.make_params())
                    params[f'{prefix}amplitude'].set(value=height, min=0)
                    params[f'{prefix}center'].set(value=pos)
                    params[f'{prefix}sigma'].set(value=1, min=0.1)

                    result = model.fit(intensity, params, x=wavelengths)

                    ss_total = np.sum((intensity - np.mean(intensity)) ** 2)
                    ss_residual = np.sum(result.residual ** 2)
                    r_squared = 1 - (ss_residual / ss_total)
                    r2_list.append(r_squared)
                    params_list.append(result.params)
                    models.append(model)
                    print(f"Using {m + 1} peak(s): R² = {r_squared:.4f}")

                    if len(peak_positions) == 1:
                        self.best_fit = m
                        self.params_fit = params_list[-1]
                        self.best_model = models[-1]
                        break

                    if len(r2_list) == len(peak_positions):
                        if r2_list[-1] - r2_list[-2] < 0.01:
                            self.best_fit = m
                            self.params_fit = params_list[-2]
                            self.best_model = models[-2]
                        else:
                            self.best_fit = m + 1
                            self.params_fit = params_list[-1]
                            self.best_model = models[-1]

                    if len(r2_list) > 1 and r2_list[-1] - r2_list[-2] < 0.01 and r2_list[-2] > 0.9:
                        self.best_fit = m
                        self.params_fit = params_list[-2]
                        self.best_model = models[-2]
                        break

                try:
                    result = self.best_model.fit(intensity, params, x=wavelengths)
                    ss_total = np.sum((intensity - np.mean(intensity)) ** 2)
                    ss_residual = np.sum(result.residual ** 2)
                    r_squared = 1 - (ss_residual / ss_total)
                    params_data.append({
                        "Key": (x, y),
                        "R^2": r_squared
                    })
                    for param_name, param in result.params.items():
                        params_data[-1][param_name] = param.value



                    # ax_main.plot(wavelengths, result.best_fit, color='blue', linewidth=2,
                    #              label=f'R²={r_squared:.4f}')
                except Exception as e:
                    print("An error occurred during fitting:")
                    traceback.print_exc()

        if save:
            folder = f'./{filename}'
            os.makedirs(folder, exist_ok=True)
            params_df = pd.DataFrame(params_data)
            params_df.to_excel("lmfit_parameters_current.xlsx", index=False)
        pass

    def get_data_coordinate(self):
        return self.data_coordinates

    @staticmethod
    def visualize_pixel_similarity(image1, image2, coord1, coord2, patch1, patch2, ssim_value, window_size=11,
                                   title='Image 2'):
        """
        Visualizes the pixel comparison process, showing the images, patches, and SSIM value.
        """
        fig, axes = plt.subplots(1, 2, figsize=(12, 6))

        # First image
        axes[0].imshow(image1, cmap='gray', extent=[0, 64, 64, 0])
        axes[0].set_title('Image 1')
        axes[0].axis('off')
        axes[0].plot(coord1[1], coord1[0], 'ro')  # Note that matplotlib uses x, y coordinates

        # Draw rectangle around patch
        rect1 = patches.Rectangle((coord1[1] - window_size // 2, coord1[0] - window_size // 2),
                                  window_size, window_size, linewidth=1, edgecolor='r', facecolor='none')
        axes[0].add_patch(rect1)

        # Second image
        axes[1].imshow(image2, cmap='gray', extent=[0, 64, 64, 0])
        axes[1].set_title(title)
        axes[1].axis('off')
        axes[1].plot(coord2[1], coord2[0], 'ro')

        rect2 = patches.Rectangle((coord2[1] - window_size // 2, coord2[0] - window_size // 2),
                                  window_size, window_size, linewidth=1, edgecolor='r', facecolor='none')
        axes[1].add_patch(rect2)

        plt.show()

        # Display the patches and SSIM value
        fig2, axes2 = plt.subplots(1, 2, figsize=(8, 4))
        axes2[0].imshow(patch1, cmap='gray', extent=[0, 64, 64, 0])
        axes2[0].set_title('Patch from Image 1')
        axes2[0].axis('off')
        axes2[1].imshow(patch2, cmap='gray', extent=[0, 64, 64, 0])
        axes2[1].set_title('Patch from Image 2')
        axes2[1].axis('off')
        plt.suptitle(f'SSIM between patches: {ssim_value:.4f}', fontsize=16)
        plt.show()

    @staticmethod
    def are_pixels_similar(image1, image2, coord1, coord2, window_size=7, threshold=0.5):
        """
        Determines if two pixels in different images are similar based on the SSIM of patches around them.

        Parameters:
        - mes1, mes2: Two set of measurements that has been done.
        - coord1, coord2: Tuples (y, x) representing the coordinates of the pixels in image1 and image2.
        - window_size: Size of the square window around the pixel to compute SSIM.
        - threshold: SSIM threshold above which pixels are considered similar.

        Returns:
        - ssim_value: The computed SSIM value.
        - is_similar: True if SSIM between patches is above the threshold, False otherwise.
        - patch1, patch2: The patches extracted from image1 and image2.
        """
        y1, x1 = coord1
        y2, x2 = coord2
        half_window = window_size // 2

        # image1 = self.data_dict[mes1].get_live_scan()
        # image2 = self.data_dict[mes2].get_live_scan()

        patch1 = image1[max(0, y1 - half_window): y1 + half_window + 1,
                 max(0, x1 - half_window): x1 + half_window + 1]
        patch2 = image2[max(0, y2 - half_window): y2 + half_window + 1,
                 max(0, x2 - half_window): x2 + half_window + 1]

        min_rows = min(patch1.shape[0], patch2.shape[0])
        min_cols = min(patch1.shape[1], patch2.shape[1])
        patch1 = patch1[:min_rows, :min_cols]
        patch2 = patch2[:min_rows, :min_cols]

        ssim_value = ssim(patch1, patch2,
                          data_range=patch1.max() - patch1.min(),
                          channel_axis=-1 if patch1.ndim == 3 else None)

        is_similar = ssim_value >= threshold
        # CondAns.visualize_pixel_similarity(image1, image2, coord1, coord2, patch1, patch2, ssim_value,
        # window_size=window_size)
        return ssim_value, is_similar, patch1, patch2

    @staticmethod
    def map_pixels(img1, img2, window_size=11, search_radius=15):
        """
        Find pixel correspondences between two images using local SSIM comparison

        Args:
            img1 (numpy.ndarray): First image (2D array)
            img2 (numpy.ndarray): Second image (2D array)
            window_size (int): Odd number size of the comparison window
            search_radius (int): Search radius in pixels around original position

        Returns:
            correspondence_map (numpy.ndarray): Array of shape (H, W, 2) containing
            corresponding [x,y] coordinates in img2 for each pixel in img1
        """

        assert img1.ndim == 2 and img2.ndim == 2, "Images must be 2D arrays"
        assert img1.shape == img2.shape, "Images must have the same dimensions"
        assert window_size % 2 == 1, "Window size must be odd"

        global_min = min(img1.min(), img2.min())
        global_max = max(img1.max(), img2.max())
        data_range = global_max - global_min
        result_dict = dict()

        pad = window_size // 2
        height, width = img1.shape

        img1_padded = np.pad(img1, pad, mode='reflect')
        img2_padded = np.pad(img2, pad, mode='reflect')

        correspondence_map = np.zeros((height, width, 2), dtype=np.int32)

        offsets = [(di, dj) for di in range(-search_radius, search_radius + 1)
                   for dj in range(-search_radius, search_radius + 1)]

        for i in range(height):
            for j in range(width):
                pi, pj = i + pad, j + pad

                window1 = img1_padded[pi - pad:pi + pad + 1, pj - pad:pj + pad + 1]

                best_score = -np.inf
                best_pos = (i, j)

                min_i = max(pad, pi - search_radius)
                max_i = min(img2_padded.shape[0] - pad, pi + search_radius + 1)
                min_j = max(pad, pj - search_radius)
                max_j = min(img2_padded.shape[1] - pad, pj + search_radius + 1)

                for x in range(min_i, max_i):
                    for y in range(min_j, max_j):
                        window2 = img2_padded[x - pad:x + pad + 1, y - pad:y + pad + 1]

                        score = ssim(window1, window2,
                                     data_range=data_range,
                                     gaussian_weights=True,
                                     win_size=window_size,
                                     use_sample_covariance=False)

                        if score > best_score:
                            best_score = score
                            best_pos = (x - pad, y - pad)

                correspondence_map[i, j] = best_pos

        for i in range(height):
            for j in range(width):
                result_dict[(i, j)] = correspondence_map[i][j]

        return result_dict


    @staticmethod
    def fit_lorentzian_spectrum(x, y, num_peaks=1, model_func=VoigtModel):
        """
        Fits the given spectrum using a sum of Lorentzian functions.

        Parameters:
            x (numpy.ndarray): The x-axis data (e.g., wavelength, energy, etc.).
            y (numpy.ndarray): The y-axis data (intensity).
            num_peaks (int): Number of Lorentzian peaks to fit.
            model_func (class): The model class to use for each peak (e.g., VoigtModel, GaussianModel).

        Returns:
            lmfit.model.ModelResult: The fitting result.
        """
        composite_model = None
        params = None

        for i in range(num_peaks):
            model = model_func(prefix=f'p{i}_')

            if composite_model is None:
                composite_model = model
            else:
                composite_model += model

            if i == 0:
                amp_guess = 2000
                center_guess = 800
                sigma_guess = 10
            elif i == 1:
                amp_guess = 2000
                center_guess = 900
                sigma_guess = 1
            elif i == 2:
                amp_guess = 2000
                center_guess = 970
                sigma_guess = 1

            model_params = model.make_params()
            if params is None:
                params = model_params
            else:
                params.update(model_params)

            params[f'p{i}_amplitude'].set(value=amp_guess, min=0)
            params[f'p{i}_center'].set(value=center_guess, min=x.min(), max=x.max())
            params[f'p{i}_sigma'].set(value=sigma_guess, min=0)

        result = composite_model.fit(y, params, x=x)
        return result

    @staticmethod
    def __plot_image_with_rect(ax, image_data, coord, title):
        """
        Plot a grayscale image on the given axis with a red rectangle
        marking the coordinate.

        Parameters:
            ax (matplotlib.axes.Axes): Axis to plot the image.
            image_data (numpy.ndarray): 2D image array.
            coord (tuple): (row, col) coordinate. If None, uses (0, 0).
            title (str): Axis title.
        """
        if coord == (0, 0):
            ax.imshow(image_data, extent=(0, image_data.shape[1], image_data.shape[0], 0),
                      cmap='gray')
            ax.set_title(title)
            ax.set_xticks([])
            ax.set_yticks([])
            ax.grid(False)
        else:
            ax.imshow(image_data, extent=(0, image_data.shape[1], image_data.shape[0], 0),
                      cmap='gray')
            rect = patches.Rectangle((coord[1], coord[0]), 1, 1,
                                     linewidth=1, edgecolor='r', facecolor='none')
            ax.add_patch(rect)
            ax.set_title(title)
            ax.set_xticks([])
            ax.set_yticks([])
            ax.grid(False)

        return coord, ax
