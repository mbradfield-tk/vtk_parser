import pyvista as pv
import numpy as np
from pathlib import Path

vtk_directory = "/Users/eqb7202/Library/CloudStorage/OneDrive-Takeda/Documents/Projects/TAK360/Step9_crystalization/CompartmentModel/VTK/surfaceFields/surfaceFields_20000.vtk"

model_path = "/Users/eqb7202/Library/CloudStorage/OneDrive-Takeda/Documents/Projects/TAK360/Step9_crystallization/MStar/"
# create path to vti files directory
vti_path = Path(model_path) / "vti"
vtk_path = Path(model_path) / "vtk"
vtk_path.mkdir(exist_ok=True)
print(f"Converting VTI files in {vti_path} to legacy VTK format in {vtk_path}")

RENAME = {"Energy Dissipation Rate (W/kg)" : "epsilon",
          "Velocity Vector (m/s)" : "U",
          "Velocity Magnitude (m/s)" : "mag(U)",
        #   "" : "nut",
        #   "" : "V",
        #   "" : "k",
          }


def print_arrays(ds: pv.DataSet, label: str = ""):
    """Print array names for point, cell, and field data in a dataset."""
    def _summarize(dsa, assoc: str):
        names = list(dsa.keys())
        print(f"[{label}] {assoc} arrays ({len(names)}):")
        if not names:
            print("   —")
            return
        for name in names:
            arr = dsa[name]
            # Number of components: 1 for scalars, >1 for vectors/tensors
            ncomp = 1 if arr.ndim == 1 else arr.shape[1]
            ntuples = arr.shape[0]
            print(f"   - {name}  (components={ncomp}, tuples={ntuples}, dtype={arr.dtype})")

    _summarize(ds.point_data, "POINT")
    _summarize(ds.cell_data,  "CELL")
    _summarize(ds.field_data, "FIELD")
    print()


# convert VTI files to legacy VTK format
for f in sorted(vti_path.glob("*.vti")):
    try:
        ds = pv.read(f)                          # auto-selects vtkXMLImageDataReader

        print_arrays(ds, label="BEFORE (VTI)")

        # rename field variables to match compartment model format
        for old, new in RENAME.items():
            if old in ds.point_data:
                ds.rename_array(old, new, preference="point")
            if old in ds.cell_data:
                ds.rename_array(old, new, preference="cell")
            if old in ds.field_data:
                ds.rename_array(old, new, preference="field")

        out_path = vtk_path / (f.stem + ".vtk")
        ds.save(out_path)     # writes legacy .vtk

        ds_written = pv.read(out_path)  # sanity check  [1](https://docs.pyvista.org/api/core/_autosummary/pyvista.dataset.rename_array)
        print_arrays(ds_written, label="AFTER  (VTK)")

    except Exception as e:
        print(f"Failed to convert {f} to VTK: {e}")
