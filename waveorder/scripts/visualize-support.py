import napari
import numpy as np
import os
import matplotlib.pyplot as plt


def plot_otf_support(
    filename,
    i,
    det_na,
    N_theta=100,
    N_phi=50,
    top_cmap="green",
    bottom_cmap="purple",
    top_azimuth_vals=None,
    bottom_azimuth_vals=None,
):
    # check azimuth values
    if top_azimuth_vals is None:
        top_azimuth_vals = np.linspace(0, 2, N_phi) % 1.0
    else:
        assert len(top_azimuth_vals) == N_phi
    if bottom_azimuth_vals is None:
        bottom_azimuth_vals = np.linspace(0, 2, N_phi) % 1.0
    else:
        assert len(bottom_azimuth_vals) == N_phi

    # key points (transverse, axial) coordinates
    points = np.array(
        [
            [0, 0],
            [det_na - i, (1 - i**2) ** 0.5 - (1 - det_na**2) ** 0.5],
            [det_na + i, (1 - i**2) ** 0.5 - (1 - det_na**2) ** 0.5],
            [2 * i, 0],
        ]
    )

    # arc centers
    centers = np.array(
        [
            [-i, (1 - i**2) ** 0.5],
            [det_na, -((1 - det_na**2) ** 0.5)],
            [i, (1 - i**2) ** 0.5],
        ]
    )

    # angles of arcs
    thetas = []
    for j, center in enumerate(centers):
        start_point = points[j]
        end_point = points[j + 1]

        theta_start = np.arctan2(start_point[1] - center[1], start_point[0] - center[0])
        theta_end = np.arctan2(end_point[1] - center[1], end_point[0] - center[0])

        thetas.append((theta_start, theta_end))

    # compute final points
    arc_lengths = [np.abs(theta[1] - theta[0]) for theta in thetas]
    total_arc_length = np.sum(arc_lengths)

    theta_coords = [
        np.linspace(
            theta[0],
            theta[1],
            np.int8(np.floor(N_theta * arc_length / total_arc_length)),
        )
        for theta, arc_length in zip(thetas, arc_lengths)
    ]

    xz_points = []
    for j, center in enumerate(centers):
        for theta_coord in theta_coords[j]:
            x = center[0] + np.cos(theta_coord)
            y = center[1] + np.sin(theta_coord)
            xz_points.append([x, y])
    xz_points = np.array(xz_points)

    phi = np.linspace(0, 2 * np.pi, N_phi, endpoint=False)

    # Compute 3D points
    points_3d = np.zeros((N_phi, xz_points.shape[0], 3))
    faces = []
    for i, xz_point in enumerate(xz_points):
        for j, angle in enumerate(phi):
            points_3d[j, i, 0] = xz_point[1]
            points_3d[j, i, 1] = xz_point[0] * np.sin(angle)
            points_3d[j, i, 2] = xz_point[0] * np.cos(angle)

            next_i = i + 1
            next_j = (j + 1) % N_phi

            faces.append([(j, i), (next_j, i), (j, next_i)])
            faces.append([(next_j, i), (next_j, next_i), (j, next_i)])

    # handle indexing
    mesh = []
    for face in faces:
        try:
            ravel_face = [
                np.ravel_multi_index(vertex, (N_phi, N_theta - 1)) for vertex in face
            ]
        except:
            print(face)
        mesh.append(ravel_face)
    mesh = np.array(mesh)

    top_values = np.tile(top_azimuth_vals, (N_theta - 1, 1)).T
    bottom_values = np.tile(bottom_azimuth_vals, (N_theta - 1, 1)).T

    points_3d = points_3d.reshape(-1, 3)
    top_values = top_values.reshape(-1)
    bottom_values = bottom_values.reshape(-1)

    v = napari.Viewer()
    
    # Add negative surface first
    points_3d_copy = points_3d.copy()
    points_3d_copy[:, 0] *= -1  # flip z
    v.add_surface(
        (points_3d_copy, mesh, bottom_values),
        opacity=0.75,
        colormap=bottom_cmap,
        blending="translucent",
        shading="smooth",
    )

    v.add_surface(
        (points_3d, mesh, top_values),
        opacity=0.75,
        colormap=top_cmap,
        blending="translucent",
        shading="smooth",
    )

    v.theme = "light"
    v.dims.ndisplay = 3
    v.camera.set_view_direction(view_direction=[-0.1, -1, -1], up_direction=[1, 0, 0])
    v.camera.zoom = 250 * 2
    import pdb; pdb.set_trace()

    v.screenshot(filename)


# Main loops
output_dir = "./output"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

N_phi = 50
N_theta = 100
det_na = 0.75
ill_nas = [0.5]

my_colors = [
    ["green", "purple"],
    2 * ["red"],
    2 * ["cyan"],
]

for ill_na in ill_nas:
    for my_color in my_colors:
        plot_otf_support(
            os.path.join(output_dir, f"{my_color[0]}-{my_color[1]}-{ill_na}.png"),
            ill_na,
            det_na,
            N_theta=N_theta,
            N_phi=N_phi,
            top_cmap=my_color[0],
            bottom_cmap=my_color[1],
            top_azimuth_vals=np.ones((N_phi,)),
            bottom_azimuth_vals=np.ones((N_phi,)),
        )

    for offset in [0, 0.5]:
        plot_otf_support(
            os.path.join(output_dir, f"hsv-{offset}-{ill_na}.png"),
            ill_na,
            det_na,
            N_theta=N_theta,
            N_phi=N_phi,
            top_cmap="hsv",
            bottom_cmap="hsv",
            top_azimuth_vals=(np.linspace(0, 2, N_phi) + offset) % 1.0,
            bottom_azimuth_vals=-(np.linspace(0, 2, N_phi) + offset) % 1.0,
        )

plot_otf_support(
    os.path.join(output_dir, f"red-red-{det_na}-{det_na}.png"),
    det_na * 0.9,
    det_na,
    N_theta=N_theta,
    N_phi=N_phi,
    top_cmap="red",
    bottom_cmap="red",
    top_azimuth_vals=np.ones((N_phi,)),
    bottom_azimuth_vals=np.ones((N_phi,)),
)
