"""
vtk_parser.py - Convert an M-Star CFD volume output (.vti) to legacy VTK (.vtk).

Selects a single time step from <model_path>/out/Output/Volume/block0
(the last one by default), renames arrays to the names expected by the
compartment model (see RENAME), and writes a binary legacy VTK file.

Usage:
    python3 vtk_parser.py <model_path> [-t TIME] [-o OUTPUT_DIR]

Arguments:
    model_path          M-Star case directory (the folder containing "out/")
    -t, --time TIME     Simulation time [s] to convert; the nearest available
                        time step is used. Default: last time step
    -o, --output-dir    Output directory. Default: <model_path>/out/VTK_from_VTI

Examples:
    python3 vtk_parser.py /data2/mstar/tak360/step9/1_TAK-360_S9_142L_76rpm_PostSeed_50C
    python3 vtk_parser.py /path/to/case -t 25
    python3 vtk_parser.py /path/to/case -t 25 -o /path/to/vtk_out

Requires: pyvista (tested with 0.49). Output is legacy VTK file version 5.1.
"""
import argparse
import re
import sys
from pathlib import Path

import pyvista as pv

VTI_SUBDIR = Path("out/Output/Volume/block0")
VTK_SUBDIR = Path("out/VTK_from_VTI")
VTI_PATTERN = re.compile(r"^Volume\.(.+)\.vti$")

# only EDR and U are needed in the compartment model (MB; 2026.10.28)
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


def find_vti_files(vti_dir: Path) -> list[tuple[float, Path]]:
    """Return (time, path) pairs sorted numerically by the time encoded in the filename."""
    files = []
    for f in vti_dir.glob("Volume.*.vti"):
        m = VTI_PATTERN.match(f.name)
        try:
            files.append((float(m.group(1)), f))
        except (AttributeError, ValueError):
            print(f"Skipping unrecognized file name: {f.name}")
    # Lexicographic sort is wrong here (e.g. 9.0e+00 > 3.0e+01 as strings)
    return sorted(files, key=lambda tf: tf[0])


def select_vti(files: list[tuple[float, Path]], time: float | None) -> tuple[float, Path]:
    if time is None:
        return files[-1]
    t, f = min(files, key=lambda tf: abs(tf[0] - time))
    if abs(t - time) > 1e-3 * max(abs(time), 1.0):
        print(f"Warning: no VTI at t={time}; using nearest t={t}")
    return t, f


def convert(vti_file: Path, out_dir: Path) -> Path:
    ds = pv.read(vti_file)
    print_arrays(ds, label="BEFORE (VTI)")

    # rename field variables to match compartment model format
    for old, new in RENAME.items():
        for assoc, dsa in (("point", ds.point_data), ("cell", ds.cell_data), ("field", ds.field_data)):
            if old in dsa:
                ds.rename_array(old, new, preference=assoc)

    out_path = out_dir / (vti_file.stem + ".vtk")
    ds.save(out_path)  # .vtk extension -> legacy writer

    print_arrays(pv.read(out_path), label="AFTER  (VTK)")
    return out_path


def main():
    parser = argparse.ArgumentParser(description="Convert an M-Star VTI volume file to legacy VTK.")
    parser.add_argument("model_path", type=Path, help="M-Star case directory (contains out/Output/Volume/block0)")
    parser.add_argument("-t", "--time", type=float, default=None,
                        help="Simulation time to convert (nearest match). Default: last time step")
    parser.add_argument("-o", "--output-dir", type=Path, default=None,
                        help=f"Output directory. Default: <model_path>/{VTK_SUBDIR}")
    args = parser.parse_args()

    vti_dir = args.model_path / VTI_SUBDIR
    out_dir = args.output_dir or args.model_path / VTK_SUBDIR

    files = find_vti_files(vti_dir)
    if not files:
        sys.exit(f"No Volume.*.vti files found in {vti_dir}")

    t, vti_file = select_vti(files, args.time)
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Converting {vti_file.name} (t={t}) -> {out_dir}")
    print(f"Wrote {convert(vti_file, out_dir)}")


if __name__ == "__main__":
    main()
