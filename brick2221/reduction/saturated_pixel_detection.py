#!/usr/bin/env python
"""
Robust saturated pixel detection for merged JWST images.

Saturated pixels in merged images (not raw JWST science images) have special
characteristics:
- Negative or very low values at the saturation core
- Surrounded by pixels with high values (blooming/diffraction artifacts)
- May not have DQ flags if this is a processed/merged image

This module provides flexible detection methods for various scenarios.

Author: GitHub Copilot
Date: February 4, 2026
"""

import numpy as np
from scipy.ndimage import binary_dilation, label, binary_erosion
from scipy.ndimage import generic_filter
from typing import Optional, Tuple


def detect_saturated_pixels_merged(
    data: np.ndarray,
    low_threshold: float = 0.0,
    high_percentile: float = 99.5,
    neighborhood_size: int = 5,
    dilation_iterations: int = 2,
    min_region_size: int = 3,
    verbose: bool = True,
) -> np.ndarray:
    """
    Detect saturated pixels in merged/processed images.
    
    Strategy: Saturated pixels in merged images have negative or low values
    at the core, surrounded by high-value pixels (blooming artifacts).
    
    Parameters
    ----------
    data : np.ndarray
        Image data (2D array)
    low_threshold : float
        Threshold for detecting low/negative pixel cores (default: 0.0)
    high_percentile : float
        Percentile for defining "high values" in surroundings (default: 99.5)
    neighborhood_size : int
        Size of neighborhood to check around low pixels (default: 5)
    dilation_iterations : int
        Number of dilation iterations to expand mask (default: 2)
    min_region_size : int
        Minimum number of connected pixels for a valid saturated region
    verbose : bool
        Print diagnostic information
        
    Returns
    -------
    mask : np.ndarray
        Boolean mask where True = saturated
    """
    # Step 1: Find pixels below threshold (potential saturation cores)
    low_mask = data < low_threshold
    
    if verbose:
        print(f"Found {low_mask.sum()} pixels below {low_threshold}")
    
    # Step 2: Find high-value pixels (blooming/artifacts)
    high_threshold = np.nanpercentile(data, high_percentile)
    high_mask = data > high_threshold
    
    if verbose:
        print(f"High value threshold (p{high_percentile}): {high_threshold:.2f}")
        print(f"Found {high_mask.sum()} high-value pixels")
    
    # Step 3: For each low pixel, check if it has high-value neighbors
    # This identifies saturation cores surrounded by blooming
    def has_high_neighbors(values):
        """Check if the center pixel is low and has high-value neighbors."""
        center = values[len(values) // 2]
        # Center is low, and at least one neighbor is high
        return center < low_threshold and np.any(values > high_threshold)
    
    saturation_mask = generic_filter(
        data,
        has_high_neighbors,
        size=neighborhood_size,
        mode='constant',
        cval=np.nan,
    ).astype(bool)
    
    # Step 4: Also include any negative pixels (clear saturation indicators)
    saturation_mask |= (data < 0)
    
    if verbose:
        print(f"After neighborhood check: {saturation_mask.sum()} saturated pixels")
    
    # Step 5: Remove small isolated regions (noise)
    if min_region_size > 0:
        labeled, n_regions = label(saturation_mask)
        region_sizes = np.bincount(labeled.ravel())
        
        # Keep only regions with sufficient pixels
        small_regions = region_sizes < min_region_size
        small_regions[0] = False  # Don't remove background
        
        saturation_mask[np.isin(labeled, np.where(small_regions)[0])] = False
        
        if verbose:
            print(f"Removed {n_regions - (region_sizes >= min_region_size).sum()} small regions")
            print(f"After cleaning: {saturation_mask.sum()} saturated pixels")
    
    # Step 6: Dilate to capture full extent of blooming
    if dilation_iterations > 0:
        original_count = saturation_mask.sum()
        saturation_mask = binary_dilation(
            saturation_mask,
            structure=np.ones((3, 3)),
            iterations=dilation_iterations,
        )
        
        if verbose:
            print(f"After {dilation_iterations} dilation iterations: {saturation_mask.sum()} pixels")
            print(f"  (added {saturation_mask.sum() - original_count} pixels)")
    
    return saturation_mask


def detect_saturated_pixels_dq(
    data: np.ndarray,
    dq: np.ndarray,
    saturated_flag: int = 2,  # JWST SATURATED flag
    dilation_iterations: int = 2,
    verbose: bool = True,
) -> np.ndarray:
    """
    Detect saturated pixels using DQ (Data Quality) flags.
    
    For raw JWST images with DQ extensions.
    
    Parameters
    ----------
    data : np.ndarray
        Image data
    dq : np.ndarray
        Data quality array
    saturated_flag : int
        DQ flag value for saturation (default: 2 for JWST)
    dilation_iterations : int
        Number of dilation iterations
    verbose : bool
        Print diagnostic information
        
    Returns
    -------
    mask : np.ndarray
        Boolean mask where True = saturated
    """
    saturation_mask = (dq & saturated_flag) > 0
    
    if verbose:
        print(f"Found {saturation_mask.sum()} saturated pixels from DQ flags")
    
    # Also check for NaN values
    saturation_mask |= ~np.isfinite(data)
    
    if verbose and (saturation_mask.sum() > (dq & saturated_flag).sum()):
        print(f"Added {saturation_mask.sum() - (dq & saturated_flag).sum()} NaN/Inf pixels")
    
    # Dilate to capture blooming
    if dilation_iterations > 0:
        original_count = saturation_mask.sum()
        saturation_mask = binary_dilation(
            saturation_mask,
            structure=np.ones((3, 3)),
            iterations=dilation_iterations,
        )
        
        if verbose:
            print(f"After {dilation_iterations} dilation iterations: {saturation_mask.sum()} pixels")
            print(f"  (added {saturation_mask.sum() - original_count} pixels)")
    
    return saturation_mask


def detect_saturated_pixels_threshold(
    data: np.ndarray,
    threshold: Optional[float] = None,
    threshold_percentile: float = 99.99,
    dilation_iterations: int = 2,
    verbose: bool = True,
) -> np.ndarray:
    """
    Detect saturated pixels using a simple threshold.
    
    For images without DQ flags, using data value thresholding.
    
    Parameters
    ----------
    data : np.ndarray
        Image data
    threshold : float, optional
        Manual saturation threshold (if None, use percentile)
    threshold_percentile : float
        Percentile to use for automatic threshold (default: 99.99)
    dilation_iterations : int
        Number of dilation iterations
    verbose : bool
        Print diagnostic information
        
    Returns
    -------
    mask : np.ndarray
        Boolean mask where True = saturated
    """
    if threshold is None:
        threshold = np.nanpercentile(data, threshold_percentile)
        
        if verbose:
            print(f"Auto threshold (p{threshold_percentile}): {threshold:.2f}")
    
    saturation_mask = data >= threshold
    
    if verbose:
        print(f"Found {saturation_mask.sum()} pixels above threshold")
    
    # Dilate to capture blooming
    if dilation_iterations > 0:
        original_count = saturation_mask.sum()
        saturation_mask = binary_dilation(
            saturation_mask,
            structure=np.ones((3, 3)),
            iterations=dilation_iterations,
        )
        
        if verbose:
            print(f"After {dilation_iterations} dilation iterations: {saturation_mask.sum()} pixels")
            print(f"  (added {saturation_mask.sum() - original_count} pixels)")
    
    return saturation_mask


def detect_saturated_pixels_with_error(
    data: np.ndarray,
    error: np.ndarray,
    min_nonzero_neighbors: int = 50,
    max_saturation_region_size: int = 10000,
    dilation_iterations: int = 2,
    verbose: bool = True,
) -> np.ndarray:
    """
    Detect saturated pixels in merged images using data and error extensions.
    
    Strategy for merged images:
    - If data=0 AND error=0, pixel is saturated or missing
    - Distinguish footprint edges from saturated stars by checking:
      * Saturated stars: small isolated clusters with nearby non-zero pixels
      * Footprint edges: large contiguous zero regions
    
    Parameters
    ----------
    data : np.ndarray
        Image data (2D array)
    error : np.ndarray
        Error/uncertainty array (2D array)
    min_nonzero_neighbors : int
        Minimum number of non-zero neighbors (in 65x65 window) for a zero pixel
        to be considered a saturated star rather than footprint edge (default: 50)
    max_saturation_region_size : int
        Maximum size of connected zero region to be considered saturation
        (larger regions are footprint edges) (default: 10000 pixels)
    dilation_iterations : int
        Number of dilation iterations to expand saturation mask (default: 2)
    verbose : bool
        Print diagnostic information
        
    Returns
    -------
    mask : np.ndarray
        Boolean mask where True = saturated (and not outside footprint)
    """
    # Step 1: Find pixels with data=0 and error=0
    zero_mask = (data == 0) & (error == 0)
    zero_mask |= ~np.isfinite(data)
    zero_mask |= ~np.isfinite(error)
    
    if verbose:
        print(f"Found {zero_mask.sum()} pixels with data=0 and error=0")
    
    # Step 2: Count non-zero neighbors to distinguish saturated stars from footprint edges
    # Saturated stars should have non-zero pixels nearby
    # Footprint edges have almost all zero neighbors
    def count_nonzero_neighbors(region):
        """Count how many pixels in region are non-zero."""
        center = region[len(region) // 2]
        # Only count neighbors if center is zero
        if center == 0:
            return np.sum(region > 0)
        else:
            return 0  # Not a zero pixel
    
    combined_zero = ((data == 0) & (error == 0)).astype(float)
    
    # Use large window to detect non-zero neighbors
    window_size = 65
    nonzero_neighbor_count = generic_filter(
        combined_zero,
        count_nonzero_neighbors,
        size=window_size,
        mode='constant',
        cval=0.0,  # Outside image counts as zero
    )
    
    # Pixels with enough non-zero neighbors are likely saturated stars
    likely_saturation = zero_mask & (nonzero_neighbor_count >= min_nonzero_neighbors)
    
    if verbose:
        print(f"Zero pixels with >={min_nonzero_neighbors} non-zero neighbors: {likely_saturation.sum()}")
    
    # Step 3: Filter by region size
    # Large contiguous zero regions are footprint edges, not saturation
    labeled, n_regions = label(zero_mask)
    
    saturation_mask = np.zeros_like(zero_mask)
    
    for region_id in range(1, n_regions + 1):
        region_mask = (labeled == region_id)
        region_size = region_mask.sum()
        
        # Check if this region has non-zero neighbors (indicates saturation, not footprint edge)
        has_nonzero_neighbors = (nonzero_neighbor_count[region_mask] >= min_nonzero_neighbors).any()
        
        # Include region if:
        # - It's small enough to be saturation, OR
        # - It has non-zero neighbors (even if large, could be blooming)
        if region_size <= max_saturation_region_size or has_nonzero_neighbors:
            saturation_mask |= region_mask
    
    if verbose:
        n_sat_regions = np.unique(labeled[saturation_mask]).size - 1  # -1 for background
        print(f"Saturated pixels (inside footprint): {saturation_mask.sum()}")
        print(f"  ({n_sat_regions} regions, max size {max_saturation_region_size} pixels)")
    
    # Step 4: Dilate saturation mask to capture blooming
    if dilation_iterations > 0:
        original_count = saturation_mask.sum()
        saturation_mask = binary_dilation(
            saturation_mask,
            structure=np.ones((3, 3)),
            iterations=dilation_iterations,
        )
        
        if verbose:
            print(f"After {dilation_iterations} dilation iterations: {saturation_mask.sum()} pixels")
            print(f"  (added {saturation_mask.sum() - original_count} pixels)")
    
    return saturation_mask


def auto_detect_saturated_pixels(
    data: np.ndarray,
    dq: Optional[np.ndarray] = None,
    error: Optional[np.ndarray] = None,
    saturated_flag: int = 2,
    is_merged: bool = False,
    verbose: bool = True,
    **kwargs,
) -> np.ndarray:
    """
    Automatically detect saturated pixels using the best available method.
    
    Decision tree:
    1. If error array is available, use data=0 & error=0 detection
    2. Else if is_merged=True, use merged image detection
    3. Else if DQ is available, use DQ flags
    4. Else use threshold-based detection
    
    Parameters
    ----------
    data : np.ndarray
        Image data
    dq : np.ndarray, optional
        Data quality array
    error : np.ndarray, optional
        Error/uncertainty array
    saturated_flag : int
        DQ flag value for saturation
    is_merged : bool
        Whether this is a merged/processed image (vs raw JWST)
    verbose : bool
        Print diagnostic information
    **kwargs
        Additional arguments passed to detection functions
        
    Returns
    -------
    mask : np.ndarray
        Boolean mask where True = saturated
    """
    if verbose:
        print("Auto-detecting saturated pixels...")
    
    if error is not None:
        if verbose:
            print("Using error extension detection (data=0 & error=0, excluding footprint edges)")
        return detect_saturated_pixels_with_error(data, error, verbose=verbose, **kwargs)
    
    elif is_merged:
        if verbose:
            print("Using merged image detection (low core + high surroundings)")
        return detect_saturated_pixels_merged(data, verbose=verbose, **kwargs)
    
    elif dq is not None:
        if verbose:
            print("Using DQ flag detection")
        return detect_saturated_pixels_dq(
            data, dq, saturated_flag=saturated_flag, verbose=verbose, **kwargs
        )
    
    else:
        if verbose:
            print("Using threshold-based detection")
        return detect_saturated_pixels_threshold(data, verbose=verbose, **kwargs)
