import iris
from .interface import FeatureExtractor
import numpy as np
from typing import Any

class WorldCoinsFeatureExtractor(FeatureExtractor):
    def __init__(self):
        super().__init__()
        # Initialize any required parameters or models here
        self.filter_bank = iris.ConvFilterBank(
            filters=[
                iris.GaborFilter(
                    kernel_size= [41, 21],
                    sigma_phi= 7.0,
                    sigma_rho= 6.13,
                    theta_degrees= 90.0,
                    lambda_phi= 28.0,
                    dc_correction= True,
                    to_fixpoints= True
                ),
                iris.GaborFilter(
                    kernel_size= [17, 21],
                    sigma_phi= 2,
                    sigma_rho= 5.86,
                    theta_degrees= 90.0,
                    lambda_phi= 8,
                    dc_correction= True,
                    to_fixpoints= True,
                )
            ],
            probe_schemas=[
                iris.RegularProbeSchema(
                    n_rows=16, n_cols=256
                ),
                iris.RegularProbeSchema(
                    n_rows=16, n_cols=256
                )
            ]
        )
        self.encoder = iris.IrisEncoder()
        
    def extract(self, normalized_iris: np.ndarray) -> np.ndarray | Any:
        """Extract features from normalized iris image."""
        # Apply filter bank to the normalized iris image
        filter_bank_output = self.filter_bank.run(
            normalization_output=normalized_iris
        )
        
        # Encode the response to get the feature vector
        encoder_output = self.encoder.run(
            response=filter_bank_output
        )
        
        return encoder_output