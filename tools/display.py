import SimpleITK as sitk
import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np

def img_show3D (img, slice= None, mask= None, scale=1, alpha = 0.3,
                maskcrop=True, padding=10):
    """Show orthogonal slices using voxel indices of the supplied full image.

    ``slice`` is an optional (x, y, z) tuple in full-image coordinates.
    With a mask, automatic slices use the centre of label 1's bounding box.
    ``maskcrop=True`` limits each view to that box plus ``padding`` voxels
    on each side, clipped to the image bounds. Axis coordinates and slice
    titles retain their full-image indices. Explicit slices may lie outside
    the box; only their in-plane display area is cropped.

    Set ``maskcrop=False`` to show the full field of view. Without a mask,
    no cropping is applied. Supply the original image and mask: coordinates
    refer to the supplied image, not to physical positions in millimetres.
    """

    if scale <= 0:
        raise ValueError("scale must be greater than 0")
    if not isinstance(padding, (int, np.integer)) or padding < 0:
        raise ValueError("padding must be a non-negative integer")
    
    margin = 0.05
    dpi = 100

    img_vec = sitk.GetArrayViewFromImage(img)
    
    if img_vec.ndim != 3:
        raise ValueError("Image must be a 3D scalar image")

    zsize, ysize, xsize = img_vec.shape

    if mask is not None:
        mask_vec = sitk.GetArrayViewFromImage(mask)

        if mask_vec.shape != img_vec.shape:
            raise ValueError("Mask and image must have the same shape")

    if mask is not None and (maskcrop or slice is None):
        stats = sitk.LabelShapeStatisticsImageFilter()
        stats.Execute(mask)

        if not stats.HasLabel(1):
            raise ValueError("Mask does not contain label 1")

        # SimpleITK bounding-box order: (x, y, z, size_x, size_y, size_z).
        x_start, y_start, z_start, x_size, y_size, z_size = (
            stats.GetBoundingBox(1)
        )

    if slice is not None:
        sagittal_slice, coronal_slice, axial_slice = slice

    elif mask is not None:
        sagittal_slice = x_start + x_size // 2
        coronal_slice = y_start + y_size // 2
        axial_slice = z_start + z_size // 2

    else:
        sagittal_slice = xsize // 2
        coronal_slice = ysize // 2
        axial_slice = zsize // 2

    for index, size, axis in zip(
        (sagittal_slice, coronal_slice, axial_slice),
        (xsize, ysize, zsize),
        "XYZ",
    ):
        if not isinstance(index, (int, np.integer)) or not 0 <= index < size:
            raise ValueError(f"{axis} slice index must be an integer in [0, {size - 1}]")

    x0, y0, z0 = 0, 0, 0
    x1, y1, z1 = xsize, ysize, zsize
    if mask is not None and maskcrop:
        x0, y0, z0 = (max(0, start - padding)
                      for start in (x_start, y_start, z_start))
        x1 = min(xsize, x_start + x_size + padding)
        y1 = min(ysize, y_start + y_size + padding)
        z1 = min(zsize, z_start + z_size + padding)

    # NumPy array order: (z, y, x). Slice indices stay in the full image.
    sagittal_img = img_vec[z0:z1, y0:y1, sagittal_slice]
    coronal_img = img_vec[z0:z1, coronal_slice, x0:x1]
    axial_img = img_vec[axial_slice, y0:y1, x0:x1]

    # Extents describe pixel edges, so voxel centres keep integer indices.
    sagittal_extent = (y0 - 0.5, y1 - 0.5, z0 - 0.5, z1 - 0.5)
    coronal_extent = (x0 - 0.5, x1 - 0.5, z0 - 0.5, z1 - 0.5)
    axial_extent = (x0 - 0.5, x1 - 0.5, y0 - 0.5, y1 - 0.5)

    if mask is not None:
        sagittal_mask = mask_vec[z0:z1, y0:y1, sagittal_slice]
        coronal_mask = mask_vec[z0:z1, coronal_slice, x0:x1]
        axial_mask = mask_vec[axial_slice, y0:y1, x0:x1]

    xsize, ysize, zsize = x1 - x0, y1 - y0, z1 - z0

    #calculate the width and height in pixel
    total_pixel_width = xsize + xsize + ysize
    max_pixel_height = max(ysize, zsize)

    #total_pixel_width = max(xsize, ysize)
    #max_pixel_height = ysize + zsize + zsize

    # Calculate the width and height in inches.
    fig_width = (1 + 3 * margin) * total_pixel_width / dpi * scale
    fig_height = (1 + margin) * max_pixel_height / dpi * scale

    figsize = (fig_width, fig_height)

    fig = plt.figure(figsize=figsize, dpi=dpi)

    #sagittal slice
    sag = fig.add_subplot(1,3,3)
    sag.imshow(sagittal_img, 
               cmap=plt.cm.Grays_r, 
               vmin= img_vec.min(), 
               vmax= img_vec.max(),
               extent=sagittal_extent,
               origin="lower"
               )

    if mask is not None:
        visible_sag_mask = np.ma.masked_where(sagittal_mask == 0, sagittal_mask)
        sag.imshow (visible_sag_mask,
                    cmap="Wistia", 
                    alpha= alpha,
                    extent=sagittal_extent,
                    origin="lower"
                    )

    sag.set_title(f"Sagittal slice\nSlice index, X = {sagittal_slice}")
    sag.set_xlabel("y-axis")
    sag.set_ylabel("z-axis")

    #coronal slice
    cor = fig.add_subplot(1,3,2)
    cor.imshow(coronal_img, 
                cmap=plt.cm.Grays_r, 
                vmin= img_vec.min(), 
                vmax= img_vec.max(),
                extent=coronal_extent,
                origin="lower"
                )

    if mask is not None:
        visible_cor_mask = np.ma.masked_where(coronal_mask == 0, coronal_mask)
        cor.imshow(visible_cor_mask, 
                    cmap="Wistia", 
                    alpha= alpha,
                    extent=coronal_extent,
                    origin="lower"
                    )

    cor.set_title(f"Coronal slice\nSlice index, Y = {coronal_slice}")
    cor.set_xlabel("x-axis")
    cor.set_ylabel("z-axis")

    #axial slice
    ax = fig.add_subplot(1,3,1)

    ax.imshow(axial_img, 
                cmap=plt.cm.Grays_r, 
                vmin= img_vec.min(), 
                vmax= img_vec.max(),
                extent=axial_extent,
                origin="lower"
                )

    if mask is not None:
        visible_ax_mask = np.ma.masked_where(axial_mask == 0, axial_mask)
        ax.imshow(visible_ax_mask, 
                    cmap="Wistia", 
                    alpha= alpha,
                    extent=axial_extent,
                    origin="lower")

    ax.set_title(f"Axial slice\nSlice index, Z = {axial_slice}")
    ax.set_xlabel("x-axis")
    ax.set_ylabel("y-axis")


    plt.show()


def img_show3D_interactive (img, slice=None, mask=None, scale=1, alpha=0.3,
                            maskcrop=True, padding=10, plane="three"):
    """Display an in-notebook CT viewer using the ipympl backend.

    Inputs match img_show3D; plane is axial, coronal, sagittal, or three.
    Enable %matplotlib widget before calling. Sliders use full-image voxel
    indices. Mask cropping and default slices use label 1's bounding box.
    Returns the displayed widget; widget.selected_point is None until a
    click selects an (x, y, z) voxel tuple. Hover coordinates are temporary.
    The minus/plus buttons move each axis by one slice within image bounds.
    """
    import ipywidgets as widgets
    from IPython.display import display

    if "ipympl" not in str(plt.get_backend()).lower():
        raise RuntimeError("Run %matplotlib widget before opening the viewer")
    if scale <= 0 or not np.isfinite(scale):
        raise ValueError("scale must be greater than 0 and finite")
    if not 0 <= alpha <= 1:
        raise ValueError("alpha must be between 0 and 1")
    if not isinstance(padding, (int, np.integer)) or padding < 0:
        raise ValueError("padding must be a non-negative integer")
    planes = ("axial", "coronal", "sagittal")
    if plane not in (*planes, "three"):
        raise ValueError("plane must be axial, coronal, sagittal, or three")

    img_vec = sitk.GetArrayFromImage(img)
    if img_vec.ndim != 3:
        raise ValueError("Image must be a 3D scalar image")
    zsize, ysize, xsize = img_vec.shape
    sizes = (xsize, ysize, zsize)
    starts = (0, 0, 0)
    ends = sizes
    centres = tuple(size // 2 for size in sizes)
    mask_vec = None
    if mask is not None:
        mask_vec = sitk.GetArrayFromImage(mask)
        if mask_vec.shape != img_vec.shape:
            raise ValueError("Mask and image must have the same shape")
        stats = sitk.LabelShapeStatisticsImageFilter()
        stats.Execute(mask)
        if not stats.HasLabel(1):
            raise ValueError("Mask does not contain label 1")
        box = stats.GetBoundingBox(1)
        starts = tuple(max(0, start - padding) for start in box[:3])
        ends = tuple(min(size, start + length + padding)
                     for size, start, length in zip(sizes, box[:3], box[3:]))
        centres = tuple(start + length // 2
                        for start, length in zip(box[:3], box[3:]))
    if slice is not None:
        if len(slice) != 3:
            raise ValueError("slice must contain (x, y, z)")
        centres = tuple(slice)
    for index, size in zip(centres, sizes):
        if not isinstance(index, (int, np.integer)) or not 0 <= index < size:
            raise ValueError("slice indices must be integers within the full image")

    plane_control = widgets.Dropdown(
        options=[("Axial", "axial"), ("Coronal", "coronal"),
                 ("Sagittal", "sagittal"), ("Three planes", "three")],
        value=plane, description="Plane")
    sliders = [widgets.IntSlider(value=int(index), min=0, max=size - 1,
                                description=axis, continuous_update=False)
               for index, size, axis in zip(centres, sizes, ("X", "Y", "Z"))]
    def move_slice(slider, step):
        slider.value = min(slider.max, max(slider.min, slider.value + step))

    axis_controls = []
    for slider in sliders:
        previous = widgets.Button(description="−", layout=widgets.Layout(width="32px"))
        following = widgets.Button(description="+", layout=widgets.Layout(width="32px"))
        previous.on_click(lambda button, slider=slider: move_slice(slider, -1))
        following.on_click(lambda button, slider=slider: move_slice(slider, 1))
        axis_controls.append(widgets.HBox([slider, previous, following]))
    alpha_control = widgets.FloatSlider(value=alpha, min=0, max=1, step=0.01,
                                        description="Alpha", disabled=mask is None)
    scale_control = widgets.BoundedFloatText(value=scale, min=0.01,
                                             max=max(100, scale), description="Scale")
    crop_control = widgets.Checkbox(value=bool(maskcrop and mask is not None),
                                    description="Crop to mask", disabled=mask is None)
    hover = widgets.Label(value="Hover: —")
    selected = widgets.Label(value="Selected: —")
    with plt.ioff():
        fig = plt.figure()
    viewer = widgets.VBox([plane_control, widgets.HBox(axis_controls),
                           widgets.HBox([alpha_control, scale_control, crop_control]),
                           hover, selected, fig.canvas])
    viewer.selected_point = None
    vmin, vmax = img_vec.min(), img_vec.max()
    axes_planes = {}

    def draw(change=None):
        fig.clear()
        axes_planes.clear()
        hover.value = "Hover: —"
        x, y, z = (slider.value for slider in sliders)
        x0, y0, z0 = starts if crop_control.value else (0, 0, 0)
        x1, y1, z1 = ends if crop_control.value else sizes
        active = planes if plane_control.value == "three" else (plane_control.value,)
        dimensions = {"axial": (x1 - x0, y1 - y0),
                      "coronal": (x1 - x0, z1 - z0),
                      "sagittal": (y1 - y0, z1 - z0)}
        width = sum(dimensions[name][0] for name in active)
        height = max(dimensions[name][1] for name in active)
        fig.set_size_inches(max(4 * len(active), width / 100) * scale_control.value,
                            max(3, height / 100) * scale_control.value)
        for control, name in zip(axis_controls, ("sagittal", "coronal", "axial")):
            control.layout.display = "" if name in active else "none"
        for position, name in enumerate(active, 1):
            if name == "axial":
                index = (z, np.s_[y0:y1], np.s_[x0:x1])
                extent = (x0 - 0.5, x1 - 0.5, y0 - 0.5, y1 - 0.5)
                horizontal, vertical, fixed, number = "x", "y", "Z", z
            elif name == "coronal":
                index = (np.s_[z0:z1], y, np.s_[x0:x1])
                extent = (x0 - 0.5, x1 - 0.5, z0 - 0.5, z1 - 0.5)
                horizontal, vertical, fixed, number = "x", "z", "Y", y
            else:
                index = (np.s_[z0:z1], np.s_[y0:y1], x)
                extent = (y0 - 0.5, y1 - 0.5, z0 - 0.5, z1 - 0.5)
                horizontal, vertical, fixed, number = "y", "z", "X", x
            ax = fig.add_subplot(1, len(active), position)
            axes_planes[ax] = name
            ax.imshow(img_vec[index], cmap=plt.cm.Grays_r, vmin=vmin, vmax=vmax,
                      origin="lower", extent=extent)
            if mask_vec is not None:
                mask_slice = mask_vec[index]
                ax.imshow(np.ma.masked_where(mask_slice == 0, mask_slice),
                          cmap="Wistia", alpha=alpha_control.value,
                          origin="lower", extent=extent)
            ax.set_title(f"{name.capitalize()} slice\nSlice index, {fixed} = {number}")
            ax.set_xlabel(f"{horizontal}-axis")
            ax.set_ylabel(f"{vertical}-axis")
        fig.tight_layout()
        fig.canvas.draw_idle()

    def point_event(event):
        if event.inaxes not in axes_planes or event.xdata is None or event.ydata is None:
            if event.name == "motion_notify_event":
                hover.value = "Hover: —"
            return
        if fig.canvas.toolbar is not None and fig.canvas.toolbar.mode:
            return
        u, v = int(np.floor(event.xdata + 0.5)), int(np.floor(event.ydata + 0.5))
        x, y, z = (slider.value for slider in sliders)
        name = axes_planes[event.inaxes]
        point = {"axial": (u, v, z), "coronal": (u, y, v),
                 "sagittal": (x, u, v)}[name]
        if not all(0 <= index < size for index, size in zip(point, sizes)):
            return
        if event.name == "button_press_event" and event.button == 1:
            viewer.selected_point = point
            selected.value = f"Selected (x, y, z): {point}"
        elif event.name == "motion_notify_event":
            hover.value = f"Hover (x, y, z): {point}"

    for control in [plane_control, *sliders, alpha_control, scale_control, crop_control]:
        control.observe(draw, names="value")
    fig.canvas.mpl_connect("motion_notify_event", point_event)
    fig.canvas.mpl_connect("button_press_event", point_event)
    draw()
    display(viewer)
    return viewer


def img_show2D(img, mask=None, scale=1, alpha = 0.3):
    dpi = 100

    if scale <= 0:
        raise ValueError("scale must be greater than 0")

    if img.GetDimension() != 2:
        raise ValueError("Image must be 2D")

    img_vec = sitk.GetArrayViewFromImage(img)

    if img_vec.ndim != 2:
        raise ValueError("Image must be a scalar image")

    # NumPy uses (y, x) order.
    ysize, xsize = img_vec.shape

    # Figure size in inches = size in pixels / dpi.
    fig_width = xsize * scale / dpi
    fig_height = ysize * scale / dpi

    fig, ax = plt.subplots(
        figsize=(fig_width, fig_height),
        dpi=dpi
    )

    ax.imshow(
        img_vec,
        cmap=plt.cm.Grays_r,
        vmin=img_vec.min(),
        vmax=img_vec.max(),
        origin="lower"
    )

    if mask is not None:
        mask_vec = sitk.GetArrayViewFromImage(mask)

        if mask_vec.shape != img_vec.shape:
            raise ValueError("Mask and image must have the same shape")

        visible_mask = np.ma.masked_where(mask_vec == 0, mask_vec)

        ax.imshow(
            visible_mask,
            cmap="Wistia",
            alpha= alpha,
            origin="lower"
        )

    ax.set_title("2D image")
    ax.set_xlabel("x-axis (pixel index)")
    ax.set_ylabel("y-axis (pixel index)")

    plt.show()


def window(img, window_width = 425, window_center = 50):
    window_width = window_width
    window_center = window_center

    window_minimum = window_center - window_width / 2
    window_maximum = window_center + window_width / 2

    windowed_img = sitk.IntensityWindowing(
        img,
        windowMinimum = window_minimum,
        windowMaximum = window_maximum,
        outputMinimum=0.0,
        outputMaximum=255.0,
    )

    return windowed_img
