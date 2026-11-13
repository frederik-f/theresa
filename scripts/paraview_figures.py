# SPDX-License-Identifier: GPL-3.0-or-later
"""Model figures rendered by ParaView from the OGS result files.

Run with ``pvpython``; ``scripts/render_figures.py`` is the usual entry point.
All figures share camera, palette and clip.

``geometry``
    the four material regions, smooth-shaded, outer regions translucent.
``mesh``
    the same view with element edges, opaque.
``field``
    temperature at a given time from a ``direct`` run.
"""

import sys
from pathlib import Path

from paraview.simple import (
    Calculator,
    Clip,
    ColorBy,
    CreateRenderView,
    ExtractSurface,
    GetColorTransferFunction,
    GetScalarBar,
    PVDReader,
    Render,
    SaveScreenshot,
    Show,
    Threshold,
    XMLUnstructuredGridReader,
)

try:  # ParaView >= 6.0
    from paraview.simple import SurfaceNormals
except ImportError:  # ParaView 5.x
    from paraview.simple import GenerateSurfaceNormals as SurfaceNormals


ARGS = sys.argv[1:]


def argument(flag: str, default: str) -> str:
    """Value of ``--flag`` on the command line, or ``default``.

    :param flag: the switch including its dashes.
    :param default: value if the switch is absent.
    """
    return ARGS[ARGS.index(flag) + 1] if flag in ARGS else default


MODE = argument("--mode", "geometry")
SOURCE = Path(argument("--source", ""))
OUT = Path(argument("--out", "/tmp/paraview_out.png"))
DRIFT_Z = float(argument("--drift-z", "-500.0"))
WINDOW = float(argument("--window", "6.5"))
SAMPLES = int(argument("--samples", "24"))
TIME_YEARS = float(argument("--time", "6.3"))
SECONDS_PER_YEAR = 365.25 * 24 * 3600.0


#: Path tracing (OSPRay) per figure mode.
RAY_TRACING = argument("--ray-tracing", "off") == "on"

#: Region colours of all figures.
COLOURS = {
    0: (0.702, 0.694, 0.676),  # canister
    1: (0.959, 0.756, 0.449),  # backfill
    2: (0.887, 0.883, 0.857),  # lining
    3: (0.633, 0.743, 0.865),  # host rock
}

INK = [0.04, 0.04, 0.04]

LABELS = {
    0: "canister (cast iron)",
    1: "backfill (bentonite)",
    2: "liner (concrete)",
    3: "host rock (claystone)",
}
#: Region opacities on a white background.
OPACITY = {0: 1.0, 1: 0.72, 2: 0.55, 3: 0.32}


def ray_traced_view(*, ray_traced: bool = True):
    """Render view on a white background.

    :param ray_traced: use the OSPRay path tracer.
    :returns: the view.
    """
    view = CreateRenderView()
    view.ViewSize = [1600, 1000]
    # White backplate; the environment does not light the scene.
    view.UseColorPaletteForBackground = 0
    view.Background = [1.0, 1.0, 1.0]
    view.Backgroundmode = "Backplate"
    view.UseGradientEnvironmentalBG = 0
    view.OrientationAxesVisibility = 0
    # Path tracer for solid figures, rasteriser for element edges.
    view.EnableRayTracing = 1 if ray_traced else 0
    if ray_traced:
        view.BackEnd = "OSPRay pathtracer"
        view.Denoise = 1
        view.SamplesPerPixel = SAMPLES
        view.Shadows = 1
        view.AmbientSamples = 4
        view.LightScale = 1.0
    return view


def place_camera(view) -> None:
    """Three-quarter view, panned so that the model sits left of the legend."""
    reach = 1.55 * WINDOW
    position = [-reach, -1.15 * reach, DRIFT_Z + 0.70 * reach]
    focal = [0.22 * WINDOW, 0.22 * WINDOW, DRIFT_Z]
    up = [0.0, 0.0, 1.0]

    direction = [focal[i] - position[i] for i in range(3)]
    right = [
        direction[1] * up[2] - direction[2] * up[1],
        direction[2] * up[0] - direction[0] * up[2],
        direction[0] * up[1] - direction[1] * up[0],
    ]
    norm = sum(component * component for component in right) ** 0.5
    pan = 0.30 * WINDOW
    shift = [component / norm * pan for component in right]

    view.CameraPosition = [position[i] + shift[i] for i in range(3)]
    view.CameraFocalPoint = [focal[i] + shift[i] for i in range(3)]
    view.CameraViewUp = up
    view.CameraParallelProjection = 0


def near_field(reader):
    """Clip the 1200 m column down to a block around the emplacement level."""
    clip = Clip(Input=reader)
    clip.ClipType = "Box"
    clip.Invert = 1
    # Exact box clip.
    clip.Exact = 1
    clip.Crinkleclip = 0
    clip.ClipType.Position = [0.0, 0.0, DRIFT_Z - 0.5 * WINDOW]
    clip.ClipType.Length = [WINDOW, WINDOW, WINDOW]
    return clip


def region(block, material: int):
    """The cells of one material."""
    part = Threshold(Input=block)
    part.Scalars = ["CELLS", "MaterialIDs"]
    part.LowerThreshold = material - 0.5
    part.UpperThreshold = material + 0.5
    return part


def smooth(source):
    """Averaged vertex normals; the feature angle keeps edges sharp."""
    surface = ExtractSurface(Input=source)
    normals = SurfaceNormals(Input=surface)
    normals.FeatureAngle = 30.0
    return normals


def categorical_map():
    """Categorical colour map on ``MaterialIDs`` with region labels.

    Opacity is set per representation.
    """
    transfer = GetColorTransferFunction("MaterialIDs")
    transfer.InterpretValuesAsCategories = 1
    transfer.AnnotationsInitialized = 1
    transfer.Annotations = [
        value for material in sorted(COLOURS) for value in (str(material), LABELS[material])
    ]
    transfer.IndexedColors = [
        channel for material in sorted(COLOURS) for channel in COLOURS[material]
    ]
    return transfer


def show_legend(display, view, transfer) -> None:
    """Categorical legend beside the model.

    :param display: a representation coloured by the categorical map.
    :param view: the view.
    :param transfer: the map from :func:`categorical_map`.
    """
    display.SetScalarBarVisibility(view, True)
    bar = GetScalarBar(transfer, view)
    bar.Title = ""
    bar.ComponentTitle = ""
    bar.LabelColor = INK
    bar.LabelFontSize = 24
    bar.WindowLocation = "Any Location"
    # Vertically centred at the right edge.
    bar.Position = [0.905, 0.34]
    bar.ScalarBarLength = 0.34
    bar.ScalarBarThickness = 28
    bar.DrawScalarBarOutline = 0


def render_geometry(reader, view, *, with_edges: bool) -> None:
    """The four material regions.

    :param reader: reader of a mesh with ``MaterialIDs``.
    :param view: the view.
    :param with_edges: draw element edges, opaque and without smoothing.
    """
    block = near_field(reader)
    transfer = categorical_map()
    for index, material in enumerate(sorted(COLOURS)):
        part = region(block, material)
        display = Show(part if with_edges else smooth(part), view)
        display.Representation = "Surface With Edges" if with_edges else "Surface"
        ColorBy(display, ("CELLS", "MaterialIDs"))
        # Opaque in mesh mode.
        display.Opacity = 1.0 if with_edges else OPACITY[material]
        if with_edges:
            display.EdgeColor = [0.08, 0.08, 0.08]
            display.LineWidth = 1.4
            display.Ambient = 0.45
            display.Diffuse = 0.62
        if index == 0:
            show_legend(display, view, transfer)


def render_field(reader, view) -> None:
    """Clip coloured by temperature [°C] at one time of a ``direct`` run."""
    celsius = Calculator(Input=near_field(reader))
    celsius.ResultArrayName = "Temperature"
    celsius.Function = "temperature - 273.15"

    display = Show(celsius, view)
    display.Representation = "Surface"
    ColorBy(display, ("POINTS", "Temperature"))
    transfer = GetColorTransferFunction("Temperature")
    transfer.ApplyPreset("Turbo", True)
    celsius.UpdatePipeline()
    display.RescaleTransferFunctionToDataRange(True, False)

    display.SetScalarBarVisibility(view, True)
    bar = GetScalarBar(transfer, view)
    bar.Title = "Temperature"
    bar.ComponentTitle = "[°C]"
    bar.AddRangeLabels = 0
    bar.TitleColor = INK
    bar.LabelColor = INK
    bar.TitleFontSize = 24
    bar.LabelFontSize = 22
    bar.WindowLocation = "Any Location"
    bar.Position = [0.855, 0.26]
    bar.ScalarBarLength = 0.48
    bar.ScalarBarThickness = 28


def main() -> None:
    """Read the source, render the requested mode and save the image."""
    if not SOURCE.exists():
        raise SystemExit(f"source not found: {SOURCE}")
    if SOURCE.suffix == ".pvd":
        reader = PVDReader(FileName=str(SOURCE))
    else:
        reader = XMLUnstructuredGridReader(FileName=[str(SOURCE)])
    reader.UpdatePipeline()

    view = ray_traced_view(ray_traced=RAY_TRACING)
    if SOURCE.suffix == ".pvd":
        # Nearest stored time step.
        available = list(reader.TimestepValues)
        wanted = TIME_YEARS * SECONDS_PER_YEAR
        view.ViewTime = min(available, key=lambda value: abs(value - wanted))
        print(f"time {view.ViewTime / SECONDS_PER_YEAR:.2f} a of {len(available)} steps")
    if MODE == "field":
        render_field(reader, view)
    else:
        render_geometry(reader, view, with_edges=MODE == "mesh")
    place_camera(view)
    Render(view)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    SaveScreenshot(str(OUT), view, ImageResolution=[1600, 1000])
    print(f"rendered -> {OUT}")


if __name__ == "__main__":
    main()
