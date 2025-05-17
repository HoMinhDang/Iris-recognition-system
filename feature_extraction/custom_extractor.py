import numpy as np
from typing import Any, List, Tuple, Union, Optional, Literal
from .interface import FeatureExtractor
from dataclasses import dataclass

@dataclass
class IrisTemplate:
    """Data class to hold iris template data."""
    iris_codes: List[np.ndarray]  # List of binary iris codes
    mask_codes: List[np.ndarray]  # List of binary mask codes
    iris_code_version: str = "v0.1"

def get_xy_mesh(kernel_size: Tuple[int, int]) -> Tuple[np.ndarray, np.ndarray]:
    """Get (x,y) meshgrids for a given kernel size."""
    ksize_phi_half = kernel_size[0] // 2
    ksize_rho_half = kernel_size[1] // 2

    y, x = np.meshgrid(
        np.arange(-ksize_phi_half, ksize_phi_half + 1),
        np.arange(-ksize_rho_half, ksize_rho_half + 1),
        indexing="xy",
        sparse=True,
    )
    return x, y

def rotate(x: np.ndarray, y: np.ndarray, angle: float) -> Tuple[np.ndarray, np.ndarray]:
    """Rotate a given array of relative positions (x,y) by a given angle."""
    cos_theta = np.cos(angle * np.pi / 180)
    sin_theta = np.sin(angle * np.pi / 180)

    rotx = x * cos_theta + y * sin_theta
    roty = -x * sin_theta + y * cos_theta

    return rotx, roty

def normalize_kernel_values(kernel_values: np.ndarray) -> np.ndarray:
    """Normalize the kernel values so that the square sum is 1."""
    norm_real = np.linalg.norm(kernel_values.real, ord="fro")
    if norm_real > 0:
        kernel_values.real /= norm_real

    norm_imag = np.linalg.norm(kernel_values.imag, ord="fro")
    if norm_imag > 0:
        kernel_values.imag /= norm_imag

    return kernel_values

def convert_to_fixpoint_kernelvalues(kernel_values: np.ndarray) -> np.ndarray:
    """Convert the kernel values to fix points."""
    if np.iscomplexobj(kernel_values):
        kernel_values.real = np.round(kernel_values.real * 2**15)
        kernel_values.imag = np.round(kernel_values.imag * 2**15)
    else:
        kernel_values = np.round(kernel_values * 2**15)
    return kernel_values

def polar_img_padding(img: np.ndarray, p_rows: int, p_cols: int) -> np.ndarray:
    """Apply zero-padding vertically and rotate-padding horizontally."""
    i_rows, i_cols = img.shape
    padded_image = np.zeros((i_rows + 2 * p_rows, i_cols + 2 * p_cols))

    padded_image[p_rows : i_rows + p_rows, p_cols : i_cols + p_cols] = img
    padded_image[p_rows : i_rows + p_rows, 0:p_cols] = img[:, -p_cols:]
    padded_image[p_rows : i_rows + p_rows, -p_cols:] = img[:, 0:p_cols]

    return padded_image

class GaborFilter:
    def __init__(
        self,
        kernel_size: Tuple[int, int],
        sigma_phi: float,
        sigma_rho: float,
        theta_degrees: float,
        lambda_phi: float,
        dc_correction: bool = True,
        to_fixpoints: bool = False,
    ):
        """Initialize Gabor filter parameters."""
        # Validate parameters
        if not all(k % 2 == 1 and 3 <= k <= 99 for k in kernel_size):
            raise ValueError("Kernel size must be odd numbers between 3 and 99")
        if sigma_phi < 1:
            raise ValueError("sigma_phi must be >= 1")
        if sigma_rho < 1:
            raise ValueError("sigma_rho must be >= 1")
        if not 0 <= theta_degrees < 360:
            raise ValueError("theta_degrees must be between 0 and 360")
        if lambda_phi < 2:
            raise ValueError("lambda_phi must be >= 2")
        
        self.kernel_size = kernel_size
        self.sigma_phi = sigma_phi
        self.sigma_rho = sigma_rho
        self.theta_degrees = theta_degrees
        self.lambda_phi = lambda_phi
        self.dc_correction = dc_correction
        self.to_fixpoints = to_fixpoints
        
        # Compute kernel values
        self.kernel_values = self.compute_kernel_values()
        self.kernel_norm = normalize_kernel_values(np.ones_like(self.kernel_values))

    def compute_kernel_values(self) -> np.ndarray:
        """Compute 2D Gabor filter kernel values."""
        # Convert to polar coordinates
        x, y = get_xy_mesh(self.kernel_size)
        rotx, roty = rotate(x, y, self.theta_degrees)

        # Calculate carrier and envelope
        carrier = 1j * 2 * np.pi / self.lambda_phi * rotx
        envelope = -(rotx**2 / self.sigma_phi**2 + roty**2 / self.sigma_rho**2) / 2

        # Calculate kernel values
        kernel_values = np.exp(envelope + carrier)
        kernel_values /= 2 * np.pi * self.sigma_phi * self.sigma_rho

        # Apply DC correction
        if self.dc_correction:
            g_mean = np.mean(np.real(kernel_values), axis=-1)
            correction_term_mean = np.mean(envelope, axis=-1)
            kernel_values = kernel_values - (g_mean / correction_term_mean)[:, np.newaxis] * envelope

        # Normalize and convert to fixpoints if needed
        kernel_values = normalize_kernel_values(kernel_values)
        if self.to_fixpoints:
            kernel_values = convert_to_fixpoint_kernelvalues(kernel_values)

        return kernel_values

class RegularProbeSchema:
    def __init__(
        self,
        n_rows: int,
        n_cols: int,
        boundary_rho: List[float] = [0, 0.0625],
        boundary_phi: Union[Literal["periodic-symmetric", "periodic-left"], List[float]] = "periodic-left",
        image_shape: Optional[List[int]] = None,
    ):
        """Initialize probe schema parameters."""
        if n_rows <= 1 or n_cols <= 1:
            raise ValueError("n_rows and n_cols must be > 1")
        if isinstance(boundary_rho, list) and (boundary_rho[0] + boundary_rho[1]) >= 1:
            raise ValueError("Sum of boundary_rho offsets must be < 1")
        
        self.n_rows = n_rows
        self.n_cols = n_cols
        self.boundary_rho = boundary_rho
        self.boundary_phi = boundary_phi
        self.image_shape = image_shape
        
        # Generate schema
        self.rhos, self.phis = self.generate_schema()

    def generate_schema(self) -> Tuple[np.ndarray, np.ndarray]:
        """Generate rhos and phis sampling positions."""
        rho = np.linspace(
            0 + self.boundary_rho[0], 1 - self.boundary_rho[1], self.n_rows, endpoint=True
        )

        if self.boundary_phi == "periodic-symmetric":
            phi = np.linspace(0, 1, self.n_cols, endpoint=False)
            phi = phi + (phi[1] - phi[0]) / 2
        elif self.boundary_phi == "periodic-left":
            phi = np.linspace(0, 1, self.n_cols, endpoint=False)
        else:  # List of offsets
            phi = np.linspace(
                0 + self.boundary_phi[0], 1 - self.boundary_phi[1], self.n_cols, endpoint=True
            )

        phis, rhos = np.meshgrid(phi, rho)
        rhos = rhos.flatten()
        phis = phis.flatten()

        # Verify pixel positions if image_shape is provided
        if self.image_shape is not None:
            rhos_pixel_values = rhos * self.image_shape[0]
            phis_pixel_values = phis * self.image_shape[1]
            if not all(v % 1 < 1e-10 or v % 1 > 1 - 1e-10 for v in rhos_pixel_values):
                raise ValueError(f"Choice for n_rows {self.n_rows} leads to interpolation errors")
            if not all(v % 1 < 1e-10 or v % 1 > 1 - 1e-10 for v in phis_pixel_values):
                raise ValueError(f"Choice for n_cols {self.n_cols} leads to interpolation errors")

        return rhos, phis

class ConvFilterBank:
    def __init__(
        self,
        filters: List[GaborFilter],
        probe_schemas: List[RegularProbeSchema],
        iris_code_version: str = "v0.1",
    ):
        """Initialize filter bank."""
        if len(filters) != len(probe_schemas):
            raise ValueError("Number of filters must match number of probe schemas")
        if not filters or not probe_schemas:
            raise ValueError("Filters and probe schemas cannot be empty")
        
        self.filters = filters
        self.probe_schemas = probe_schemas
        self.iris_code_version = iris_code_version

    def run(self, normalization_output: Any) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        """Apply filters to normalized iris image."""
        iris_responses = []
        mask_responses = []

        for i_filter, i_schema in zip(self.filters, self.probe_schemas):
            iris_response, mask_response = self._convolve(i_filter, i_schema, normalization_output)
            iris_responses.append(iris_response)
            mask_responses.append(mask_response)

        return iris_responses, mask_responses

    def _convolve(
        self, img_filter: GaborFilter, probe_schema: RegularProbeSchema, normalization_output: Any
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Apply convolution to normalized iris image."""
        i_rows, i_cols = normalization_output.normalized_image.shape
        k_rows, k_cols = img_filter.kernel_values.shape
        p_rows, p_cols = k_rows // 2, k_cols // 2

        n_rows, n_cols = probe_schema.n_rows, probe_schema.n_cols
        iris_response = np.zeros((n_rows, n_cols), dtype=np.complex64)
        mask_response = np.zeros((n_rows, n_cols), dtype=np.complex64)

        padded_iris = polar_img_padding(normalization_output.normalized_image, 0, p_cols)
        padded_mask = polar_img_padding(normalization_output.normalized_mask, 0, p_cols)

        for i in range(n_rows):
            for j in range(n_cols):
                # Convert probe_schema position to integer pixel position
                pos = i * n_cols + j
                r_probe = min(round(probe_schema.rhos[pos] * i_rows), i_rows - 1)
                c_probe = min(round(probe_schema.phis[pos] * i_cols), i_cols - 1)

                # Get patch from image centered at [i,j] probed pixel position
                rtop = max(0, r_probe - p_rows)
                rbot = min(r_probe + p_rows + 1, i_rows - 1)
                iris_patch = padded_iris[rtop:rbot, c_probe : c_probe + k_cols]
                mask_patch = padded_mask[rtop:rbot, c_probe : c_probe + k_cols]

                # Perform convolution at [i,j] probed pixel position
                ktop = p_rows - iris_patch.shape[0] // 2
                iris_response[i][j] = (
                    (iris_patch * img_filter.kernel_values[ktop : ktop + iris_patch.shape[0], :]).sum()
                    / iris_patch.shape[0]
                    / k_cols
                )
                mask_response[i][j] = (
                    0 if iris_response[i][j] == 0 else (mask_patch.sum() / iris_patch.shape[0] / k_cols)
                )

        # Normalize responses using scalar values
        kernel_norm_real = np.linalg.norm(img_filter.kernel_values.real, ord="fro")
        kernel_norm_imag = np.linalg.norm(img_filter.kernel_values.imag, ord="fro")
        
        if kernel_norm_real > 0:
            iris_response.real = iris_response.real / kernel_norm_real
        if kernel_norm_imag > 0:
            iris_response.imag = iris_response.imag / kernel_norm_imag
        mask_response.imag = mask_response.real

        return iris_response, mask_response

class CustomFeatureExtractor(FeatureExtractor):
    def __init__(self):
        super().__init__()
        # Create Gabor filters
        self.filters = [
            GaborFilter(
                kernel_size=(41, 21),
                sigma_phi=7.0,
                sigma_rho=6.13,
                theta_degrees=90.0,
                lambda_phi=28.0,
                dc_correction=True,
                to_fixpoints=True
            ),
            GaborFilter(
                kernel_size=(17, 21),
                sigma_phi=2.0,
                sigma_rho=5.86,
                theta_degrees=90.0,
                lambda_phi=8.0,
                dc_correction=True,
                to_fixpoints=True
            )
        ]
        
        # Create probe schemas
        self.probe_schemas = [
            RegularProbeSchema(n_rows=16, n_cols=256),
            RegularProbeSchema(n_rows=16, n_cols=256)
        ]
        
        # Create filter bank
        self.filter_bank = ConvFilterBank(
            filters=self.filters,
            probe_schemas=self.probe_schemas,
            iris_code_version="v0.1"
        )
        
    def extract(self, normalization_output: Any) -> IrisTemplate:
        """Extract features from normalized iris image."""
        # Apply filter bank
        iris_responses, mask_responses = self.filter_bank.run(normalization_output)
        
        # Binarize responses
        iris_codes = []
        mask_codes = []
        
        for iris_response, mask_response in zip(iris_responses, mask_responses):
            # Binarize iris response
            iris_code = np.stack([
                iris_response.real > 0,
                iris_response.imag > 0
            ], axis=-1)
            
            # Binarize mask
            mask_code = np.stack([
                mask_response.real > 0.9,
                mask_response.real > 0.9
            ], axis=-1)
            
            iris_codes.append(iris_code)
            mask_codes.append(mask_code)
        
        # Create template
        template = IrisTemplate(
            iris_codes=iris_codes,
            mask_codes=mask_codes,
            iris_code_version="v0.1"
        )
        
        return template
