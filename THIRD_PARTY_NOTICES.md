# THIRD PARTY NOTICES

This branch implements its Ground Engine in Python/NumPy/SciPy. No source code from the projects below is embedded unless explicitly stated.

## r-lidar/PTD

Repository: https://github.com/r-lidar/PTD

Use: algorithmic reference for Progressive TIN Densification concepts, shifted low-point seed grids, triangle distance/angle tests, progressive insertion, and spike rejection.

Code reused: none.

License note: no source file from this repository is copied into LAS-CAFIISICA.

## Ground-Extraction-From-Point-Cloud

Repository: https://github.com/fazanham/Ground-Extraction-From-Point-Cloud

License: MIT.

Use: architectural reference for coarse non-ground removal followed by TIN-based refinement and block-oriented processing.

Code reused: none.

## CSF

Repository: https://github.com/jianboqi/CSF

License: Apache-2.0 according to the upstream README.

Use: algorithmic reference for an independent cloth-simulation ground opinion.

Code reused: none. LAS-CAFIISICA currently uses an internal SciPy cloth-like raster implementation rather than bundling the upstream C++/SWIG code.

## PDAL / Open3D / PCL

Used as conceptual references for outlier filtering, HAG, KD trees, PCA/normals and point-cloud geometry.

Code reused: none.
