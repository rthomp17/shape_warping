# Code by Ondrej Biza and Skye Thompson
import copy as cp
from typing import Dict, List, Optional

import numpy as np
import plotly.graph_objects as go
from numpy.typing import NDArray
from plotly.subplots import make_subplots


# Show point clouds from dictionary, formatted as {name: pcd}
def show_pcds_plotly(
    pcds: Dict[str, NDArray],
    center: bool = False,
    axis_visible: bool = True,
    colors: Optional[Dict[str, str]] = None,
    show_legend=True,
    markers=None,
    camera=None,
    title= None,
):
    colorscales = [
        "Plotly3",
        "Viridis",
        "Blues",
        "Greens",
        "Greys",
        "Oranges",
        "Purples",
        "Reds",
    ]

    if center:
        tmp = np.concatenate(list(pcds.values()), axis=0)
        m = np.mean(tmp, axis=0)
        pcds = copy.deepcopy(pcds)
        for k in pcds.keys():
            pcds[k] = pcds[k] - m[None]

    tmp = np.concatenate(list(pcds.values()), axis=0)
    lmin = np.min(tmp)
    lmax = np.max(tmp)

    data = []
    for idx, key in enumerate(pcds.keys()):
        v = pcds[key]
        if colors is not None:
            colorscale = colors[key]
        else:
            colorscale = colorscales[idx % len(colorscales)]

        if markers is not None:
            marker = markers[key] | {"color": v[:, 2], "colorscale": colorscale}
        else:
            marker = {
                "size": 5,
                "color": v[:, 2],
                "colorscale": colorscale,
            }

        pl = go.Scatter3d(
            x=v[:, 0],
            y=v[:, 1],
            z=v[:, 2],
            marker=marker,
            mode="markers",
            name=key,
        )
        data.append(pl)

    layout = {
        "xaxis": {"visible": axis_visible, "range": [lmin, lmax]},
        "yaxis": {"visible": axis_visible, "range": [lmin, lmax]},
        "zaxis": {"visible": axis_visible, "range": [lmin, lmax]},
        "aspectratio": {"x": 1, "y": 1, "z": 1},
    }
    fig = go.Figure(data=data)
    if camera is None:
        camera = fig.layout.scene.camera
        camera.up = dict(x=0, y=1, z=0)
        camera.eye = dict(x=6.28, y=0, z=1)
    fig.update_layout(scene=layout, scene_camera=camera, showlegend=show_legend)

    if title is not None: 
        fig.update_layout(title={
        'text': title,})

    # fig.show()
    return fig

# Show meshes from two dictionaries, formatted as {name: vertices}, {name: faces}
def show_meshes_plotly(
    vertices: Dict[str, NDArray],
    faces: Dict[str, NDArray],
    center: bool = False,
    axis_visible: bool = True,
    camera: Optional[Dict[str, Dict[str, float]]] = None,
    show_legend: bool = True,
    show: bool = False,
    pcds=None,
):
    colorscales = [
        "Plotly3",
        "Viridis",
        "Blues",
        "Greens",
        "Greys",
        "Oranges",
        "Purples",
        "Reds",
    ]

    if center:
        tmp = np.concatenate(list(vertices.values()), axis=0)
        m = np.mean(tmp, axis=0)
        vertices = copy.deepcopy(vertices)
        for k in vertices.keys():
            vertices[k] = vertices[k] - m[None]

    tmp = np.concatenate(list(vertices.values()), axis=0)
    lmin = np.min(tmp)
    lmax = np.max(tmp)

    data = []
    for idx, key in enumerate(vertices.keys()):
        v = vertices[key]
        f = faces[key]
        colorscale = colorscales[idx % len(colorscales)]
        mesh = go.Mesh3d(
            x=v[:, 0],
            y=v[:, 1],
            z=v[:, 2],
            i=f[:, 0],
            j=f[:, 1],
            k=f[:, 2],
            colorscale=colorscale,
            intensity=v[:, 2],
            showscale=False,
        )
        data.append(mesh)

    if pcds is not None:
        for idx, key in enumerate(pcds.keys()):
            v = pcds[key]
            # if colors is not None:
            #     colorscale = colors[key]
            # else:
            #     colorscale = colorscales[idx % len(colorscales)]

            # if markers is not None:
            #     marker = markers[key] | {"color":v[:, 2], "colorscale": colorscale}
            # else:
            #     marker={"size": 5,
            #             "color": v[:, 2],
            #             "colorscale": colorscale,
            #                 }

            pl = go.Scatter3d(
                x=v[:, 0],
                y=v[:, 1],
                z=v[:, 2],
                # marker=marker,
                mode="markers",
                name=key,
            )
            data.append(pl)

    layout = {
        "xaxis": {"visible": axis_visible, "range": [lmin, lmax]},
        "yaxis": {"visible": axis_visible, "range": [lmin, lmax]},
        "zaxis": {"visible": axis_visible, "range": [lmin, lmax]},
        "aspectratio": {"x": 1, "y": 1, "z": 1},
    }

    fig = go.Figure(data=data)
    if camera is None:
        camera = fig.layout.scene.camera
        camera.up = dict(x=0, y=0, z=1)
        camera.eye = dict(x=1.6, y=0.6, z=0.8)
    fig.update_xaxes(showgrid=axis_visible)
    fig.update_yaxes(showgrid=axis_visible)
    fig.update_layout(
        scene=layout, scene_camera=camera, showlegend=show_legend, width=600, height=600
    )
    if show:
        fig.show()

    return fig


from plotly.subplots import make_subplots


# Show point clouds in a grid of subplots
def show_pcd_grid_plotly(
    rows,
    cols,
    pcls: Dict[str, Dict[str, NDArray]],
    names: List[str],
    subplot_titles=List[str],
    markers=None,
    camera_views: Optional[List[Dict[str, dict]]] = None,
    get_images=False,
    save_image: bool = False,
    save_path: bool = False,
):
    fig = make_subplots(
        rows=rows,
        cols=cols,
        specs=[[{"type": "scatter3d"} for i in range(cols)] for j in range(rows)],
        horizontal_spacing=0.00,
        vertical_spacing=0.00,
        subplot_titles=subplot_titles,
    )
    layout = {}
    axis_visible = True

    for row in range(rows):
        for col in range(cols):

            name = names[row * cols + col]

            tmp = np.concatenate(list(pcls[name].values()), axis=0)
            epsilon = 0.005
            lmin = np.min(tmp) - epsilon
            lmax = np.max(tmp) + epsilon

            for pcl_name in pcls[name].keys():
                if markers is not None:
                    marker = markers[name][pcl_name] | {
                        "color": np.ones_like(pcls[name][pcl_name][:, 2]) * 0.5,
                    }
                else:
                    marker = {
                        "size": 5,
                        "color": pcls[name][pcl_name][:, 2],
                        "colorscale": "viridis",
                    }
                fig.add_trace(
                    go.Scatter3d(
                        x=pcls[name][pcl_name][:, 0],
                        y=pcls[name][pcl_name][:, 1],
                        z=pcls[name][pcl_name][:, 2],
                        marker=marker,
                        mode="markers",
                        name=pcl_name,
                    ),
                    row=row + 1,
                    col=col + 1,
                )

            layout = layout | {
                "xaxis": {"visible": axis_visible, "range": [lmin, lmax]},
                "yaxis": {"visible": axis_visible, "range": [lmin, lmax]},
                "zaxis": {"visible": axis_visible, "range": [lmin, lmax]},
                "aspectratio": {"x": 1, "y": 1, "z": 1},
            }
            plot_index = row * cols + col + 1 if row * cols + col > 0 else ""
            eval(f"fig.layout.scene{plot_index}.xaxis").update(
                {"visible": axis_visible, "range": [lmin, lmax]}
            )
            eval(f"fig.layout.scene{plot_index}.yaxis").update(
                {"visible": axis_visible, "range": [lmin, lmax]}
            )
            eval(f"fig.layout.scene{plot_index}.zaxis").update(
                {"visible": axis_visible, "range": [lmin, lmax]}
            )

    fig.update_annotations(font_size=25)
    fw = go.FigureWidget(fig)

    if camera_views is not None:
        all_cameras = [
            eval(f"fw.layout.scene{i}.camera") for i in range(1, rows * cols + 1, 1)
        ]
        with fw.batch_update():
            for i in range(len(all_cameras)):
                camera = all_cameras[i]
                camera.up = camera_views[i]["up"]  # dict(x=0, y=1, z=0)
                camera.eye = camera_views[i]["eye"]  # dict(x=2.5, y=1.75, z=1)
                camera.center = camera_views[i]["center"]

    fw.update_layout(scene=layout, height=1000, width=1000)

    if save_image:
        fw.write_image(save_path)

    fw.show()
    return fw


def plotly_fig2array(fig):
    # convert Plotly fig to  an array
    fig_bytes = fig.to_image(format="png")
    buf = io.BytesIO(fig_bytes)
    img = Image.open(buf)
    return np.asarray(img)


# Turns a slider fig into a video
# Every frame should be a point cloud
def show_pcds_video_animation_plotly(
    moving_pcl_name: str,
    moving_pcl_frames: NDArray,
    static_pcls: Dict[str, NDArray],
    step_names: List[str],
    file_name: str,
):

    # Additional imports for video generation
    import moviepy.editor as mpy
    import io
    import copy as cp
    from PIL import Image

    video_length = 2

    visible = [False] * (len(moving_pcl_frames)) + [True] * len(static_pcls.keys())

    fig = go.Figure()

    fig.layout.scene.update(
        dict(
            aspectmode="cube",
            xaxis=dict(range=[-0.1, 0.1], visible=False),
            yaxis=dict(range=[-0.1, 0.1], visible=False),
            zaxis=dict(range=[-0.1, 0.1], visible=False),
            aspectratio=dict(x=1, y=1, z=0.95),
        )
    )

    camera = dict(eye=dict(x=1.25, y=0.0, z=0.0))

    fig.update_layout(scene_camera=camera)

    # Add traces, one for each slider step
    for t, moving_pcl_frame in enumerate(moving_pcl_frames):
        for part in moving_pcl_frame.keys():
            fig.add_trace(
                go.Scatter3d(
                    visible=False,
                    x=moving_pcl_frame[part][:, 0],
                    y=moving_pcl_frame[part][:, 1],
                    z=moving_pcl_frame[part][:, 2],
                    marker={
                        "size": 5,
                        "color": moving_pcl_frame[part][:, 2],
                        "colorscale": "viridis",
                    },
                    mode="markers",
                    opacity=1.0,
                    name=f"{moving_pcl_name} {part}, Step {t}",
                ),
            )

    for static_name in static_pcls.keys():
        fig.add_trace(
            go.Scatter3d(
                visible=True,
                x=static_pcls[static_name][:, 0],
                y=static_pcls[static_name][:, 1],
                z=static_pcls[static_name][:, 2],
                marker={
                    "size": 5,
                    "color": static_pcls[static_name][:, 2],
                    "colorscale": "plotly3",
                },
                mode="markers",
                opacity=1.0,
                name=static_name,
            ),
        )

    def make_frame(t):
        i = int(t * len(moving_pcl_frames))

        for j in range(len(moving_pcl_frames)):
            for part in moving_pcl_frames[j].keys():
                fig.update_traces(
                    visible=False,
                    selector=dict(name=f"{moving_pcl_name} {part}, Step {j}"),
                )

        for part in moving_pcl_frames[i].keys():
            fig.update_traces(
                visible=True, selector=dict(name=f"{moving_pcl_name} {part}, Step {i}")
            )  # These are the updates that usually are performed within Plotly go.Frame definition
        fig.update_layout(title=step_names[i])
        return plotly_fig2array(fig)

    animation = mpy.VideoClip(make_frame, duration=1)
    animation.write_gif(f"{file_name}.gif", fps=20)


# Slider animation showing a series of point clouds as frames
# Only one frame is made visible for each slider step
def show_pcds_slider_animation_plotly(
    moving_pcl_name: str,
    moving_pcl_frames: NDArray,
    static_pcls: Dict[str, NDArray],
    step_names: List[str],
):
    fig = go.Figure()
    fig.layout.scene.update(
        dict(
            aspectmode="cube",
            xaxis=dict(range=[-0.1, 0.1], visible=False),
            yaxis=dict(range=[-0.1, 0.1], visible=False),
            zaxis=dict(range=[-0.1, 0.1], visible=False),
            aspectratio=dict(x=1, y=1, z=0.95),
        )
    )

    camera = dict(eye=dict(x=1.25, y=0.0, z=0.0))

    fig.update_layout(scene_camera=camera)

    # Add traces, one for each slider step
    for t, moving_pcl_frame in enumerate(moving_pcl_frames):
        for part in moving_pcl_frame.keys():
            fig.add_trace(
                go.Scatter3d(
                    visible=False,
                    x=moving_pcl_frame[part][:, 0],
                    y=moving_pcl_frame[part][:, 1],
                    z=moving_pcl_frame[part][:, 2],
                    marker={
                        "size": 5,
                        "color": moving_pcl_frame[part][:, 2],
                        "colorscale": "viridis",
                    },
                    mode="markers",
                    opacity=1.0,
                    name=f"Source {part}, Step {t}",
                ),
            )

    for static_name in static_pcls.keys():
        fig.add_trace(
            go.Scatter3d(
                x=static_pcls[static_name][:, 0],
                y=static_pcls[static_name][:, 1],
                z=static_pcls[static_name][:, 2],
                marker={
                    "size": 5,
                    "color": static_pcls[static_name][:, 2],
                    "colorscale": "plotly3",
                },
                mode="markers",
                opacity=1.0,
                name=static_name,
            ),
        )

    fig.data[0].visible = True

    # Create and add slider
    steps = []
    for i in range(len(fig.data) - 1):
        step = dict(
            method="update",
            args=[
                {
                    "visible": [False] * (len(moving_pcl_frames))
                    + [True] * len(static_pcls.keys())
                },
                {"title": step_names[i]},
            ],
        )
        step["args"][0]["visible"][i] = True  # Toggle i'th trace to "visible"
        steps.append(step)

    sliders = [
        dict(active=10, currentvalue={"prefix": "Step: "}, pad={"t": 50}, steps=steps)
    ]

    fig.update_layout(sliders=sliders)
    return fig

# Utility function to prevent circular imports
def transform_pcd(pcd, trans, is_position: bool = True):
    n = pcd.shape[0]
    cloud = cp.deepcopy(pcd.T)
    augment = np.ones((1, n)) if is_position else np.zeros((1, n))
    cloud = np.concatenate((cloud, augment), axis=0)
    cloud = np.dot(trans.astype(np.float32), cloud)
    cloud = cloud[0:3, :].T
    return cloud

# Function for visualizing the warp optimization history
def generate_slider_viz(
    warp, static_pcls, tf_pcl, generate_animation=False, experiment_id=None
):
    best_idx = np.argmin(warp.cost_history[-1])
    best_transform_history = []
    best_transforms = []
    step_names = []

    tf2_history = []
    for transform, cost in zip(warp.transform_history, warp.cost_history):
        best_trans = transform[best_idx]
        best_transforms.append(transform_pcd(tf_pcl, best_trans.astype(float)))
        step_names.append(f"COST: {cost[best_idx]}")

    if generate_animation:
        show_pcds_video_animation_plotly(
            moving_pcl_name="Source",
            moving_pcl_frames=best_transform_history,
            static_pcls=static_pcls,
            step_names=step_names,
            file_name=experiment_id,
        )

    slider_fig = show_pcds_slider_animation_plotly(
        moving_pcl_name="Source",
        moving_pcl_frames=best_transforms,
        static_pcls=static_pcls,
        step_names=step_names,
    )
    return slider_fig

# Function for visualizing relational descriptors
# TODO: Make more general rather than teapot-specific
def visualize_teapot_relational_descriptors(part_pairs,
                                            part_pcls,
                                            pcl_labels,
                                            title=None):
    all_labeled_parts = {}
    
    part_nums = {}

    # Unpacking labels and creating a named point cloud for every part-label combination
    for part_pair in part_pairs:
        ordered_part_pair = list(part_pair.keys())
        ordered_part_pair.sort()

        for part in ordered_part_pair:
            if part in part_nums.keys():
                i = part_nums[part]
                all_labeled_parts = (
                    all_labeled_parts
                    | {
                        f"{ordered_part_pair[0]}_{ordered_part_pair[1]}_0_{part}": part_pcls[part][
                            pcl_labels[part][i] == 0
                        ]
                    }
                    | {
                        f"{ordered_part_pair[0]}_{ordered_part_pair[1]}_1_{part}": part_pcls[part][
                            pcl_labels[part][i] == 1
                        ]
                    }
                )
                part_nums[part] += 1
            else:
                part_nums[part] = 1
                all_labeled_parts = (
                    all_labeled_parts
                    | {
                        f"{ordered_part_pair[0]}_{ordered_part_pair[1]}_0_{part}": part_pcls[part][
                            pcl_labels[part][0] == 0
                        ]
                    }
                    | {
                        f"{ordered_part_pair[0]}_{ordered_part_pair[1]}_1_{part}": part_pcls[part][
                            pcl_labels[part][0] == 1
                        ]
                    }
                )

    # Hard coded colors
    colors = {
        "body_lid_0_body": "gray",
        "body_lid_1_body": "greens",
        "body_lid_0_lid": "gray",
        "body_lid_1_lid": "greens",
        "body_spout_0_body": "gray",
        "body_spout_1_body": "oranges",
        "body_spout_0_spout": "gray",
        "body_spout_1_spout": "oranges",
        "body_handle_0_body": "gray",
        "body_handle_1_body": "purples",
        "body_handle_0_handle": "gray",
        "body_handle_1_handle": "purples",
        "handle_lid_0_handle": "gray",
        "handle_lid_1_handle": "reds",
        "handle_lid_0_lid": "gray",
        "handle_lid_1_lid": "reds",
        "lid_spout_0_lid": "gray",
        "lid_spout_1_lid": "blues",
        "lid_spout_0_spout": "gray",
        "lid_spout_1_spout": "blues",
    }
    show_pcds_plotly(all_labeled_parts, colors=colors, title=title).show()

