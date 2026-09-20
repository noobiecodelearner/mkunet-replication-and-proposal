"""
Stub SimpleITK module. The real SimpleITK cannot be built on a network-isolated
training server (its CMake superbuild clones ITK from GitHub, which is blocked).
This stub exists only to satisfy `import SimpleITK as sitk` in the MK-UNet
repository's utils/utils.py. The polyp/BraTS training and testing pipelines
used in this project never actually call any SimpleITK function -- those calls
only exist in volumetric (.nii.gz) save paths unused by train_polyp.py,
train_brats.py, or train_polyp_paperconfig.py. Verified by grepping every
`sitk.*` call site in utils/utils.py before writing this stub.

If any of these functions are actually called, they raise clearly rather than
silently doing nothing.

Installation: place this file as SimpleITK.py in your environment's
site-packages directory (find it with `python -c "import site;
print(site.getsitepackages())"`), instead of `pip install simpletk`, if you
hit the same network-isolation build failure.
"""


def _unavailable(*args, **kwargs):
    raise RuntimeError(
        "SimpleITK is not installed on this server (build blocked by network "
        "isolation). This function is not used by the polyp/BraTS pipelines "
        "in this project -- if you're seeing this, something unexpected "
        "called into SimpleITK."
    )


GetImageFromArray = _unavailable
WriteImage = _unavailable
ReadImage = _unavailable
GetArrayFromImage = _unavailable
