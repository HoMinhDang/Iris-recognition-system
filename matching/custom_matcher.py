import numpy as np
from typing import Any, Tuple, Optional
from .interface import IrisMatcher
from feature_extraction.custom_extractor import IrisTemplate

def normalized_hamming_distance(
    irisbitcount: int,
    maskbitcount: int,
    norm_mean: float = 0.45,
    norm_gradient: float = 0.00005
) -> float:
    """Calculate normalized Hamming distance."""
    # Linear approximation of normalization term
    norm_hd = max(0, norm_mean - (norm_mean - irisbitcount / maskbitcount) * (norm_gradient * maskbitcount + 0.5))
    return norm_hd

def get_bitcounts(
    template_probe: IrisTemplate,
    template_gallery: IrisTemplate,
    shift: int
) -> Tuple[list, list]:
    """Get bitcounts in iris and mask codes."""
    # Compare iris codes with rotation
    irisbits = [
        np.roll(probe_code, shift, axis=1) != gallery_code
        for probe_code, gallery_code in zip(template_probe.iris_codes, template_gallery.iris_codes)
    ]
    # Get common unmasked regions
    maskbits = [
        np.roll(probe_code, shift, axis=1) & gallery_code
        for probe_code, gallery_code in zip(template_probe.mask_codes, template_gallery.mask_codes)
    ]
    return irisbits, maskbits

def count_nonmatch_bits(irisbits: list, maskbits: list) -> Tuple[int, int]:
    """Count non-matching bits in common unmasked regions."""
    # Count non-matching bits in iris codes
    irisbitcount = sum(np.sum(x & y) for x, y in zip(irisbits, maskbits))
    # Count total bits in common unmasked regions
    maskbitcount = sum(np.sum(y) for y in maskbits)
    return irisbitcount, maskbitcount

class CustomMatcher(IrisMatcher):
    def __init__(
        self,
        threshold: float = 0.5,
        rotation_shift: int = 15,
        normalize: bool = True,
        norm_mean: float = 0.45,
        norm_gradient: float = 0.00005
    ):
        """Initialize matcher parameters."""
        self.threshold = threshold
        self.rotation_shift = rotation_shift
        self.normalize = normalize
        self.norm_mean = norm_mean
        self.norm_gradient = norm_gradient
        
    def match(self, feature1: np.ndarray | Any, feature2: np.ndarray | Any) -> float:
        """Match two iris templates using Hamming distance."""
        # Validate input types
        if not isinstance(feature1, IrisTemplate) or not isinstance(feature2, IrisTemplate):
            raise ValueError("Features must be IrisTemplate objects")
            
        # Validate template shapes
        for code1, code2 in zip(feature1.iris_codes, feature2.iris_codes):
            if code1.shape != code2.shape:
                raise ValueError("Iris codes have different shapes")
                
        # Find best match over allowed rotations
        best_distance = 1.0
        for shift in [0] + [y for x in range(1, self.rotation_shift + 1) for y in (-x, x)]:
            # Get bit counts for current rotation
            irisbits, maskbits = get_bitcounts(feature1, feature2, shift)
            total_iris_bits, total_mask_bits = count_nonmatch_bits(irisbits, maskbits)
            
            # Skip if no common unmasked region
            if total_mask_bits == 0:
                continue
                
            # Calculate distance
            if self.normalize:
                distance = normalized_hamming_distance(
                    total_iris_bits,
                    total_mask_bits,
                    self.norm_mean,
                    self.norm_gradient
                )
            else:
                distance = total_iris_bits / total_mask_bits
                
            # Update best match
            if distance < best_distance:
                best_distance = distance
                
        return best_distance 