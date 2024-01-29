import plotly.graph_objects as go
import numpy as np
from plyfile import PlyData, PlyElement
import plotly.io as py

def plot_points(pts, opacities, sizes = None):
    return go.Scatter3d(
        x=pts[:,0],
        y=pts[:,1],
        z=pts[:,2],
        mode='markers',
        marker=dict(
            size=sizes,
            color=opacities,    # 根据颜色数组设置每个点的颜色
            colorscale=[[0, 'rgb(5, 10, 172)'], [.01, 'rgb(255, 255, 255)'], [1, 'rgb(178, 10, 28)']],
            colorbar_title = 'Color Bar',
            line_width=0,
            opacity=1
        )
    )


def load_ply(path):
    plydata = PlyData.read(path)

    xyz = np.stack((np.asarray(plydata.elements[0]["x"]),
                    np.asarray(plydata.elements[0]["y"]),
                    np.asarray(plydata.elements[0]["z"])),  axis=1)
    opacities = np.asarray(plydata.elements[0]["opacity"])[..., np.newaxis]
    return xyz, opacities

if __name__ == "__main__":
    xyz, opacities = load_ply(path="gaussian_5600.ply")
    opacities = opacities.squeeze()
    opacities = np.exp(opacities) / (1 + np.exp(opacities))
    xyz = xyz[opacities > 0.2]
    opacities = opacities[opacities > 0.2]

    camera = [[0.0194, -0.2965,  6.0277], [-0.4727,  0.4844,  5.9084], [-0.0458, -0.2115,  5.9443]]
    camera = np.array(camera)
    xyz = np.concatenate((xyz, camera), axis=0)
    opacities = np.concatenate((opacities, np.ones((3,))), axis=0)
    sizes = np.ones_like(opacities) * 3
    sizes[-3:] = 10
    # from pdb import set_trace; set_trace()
    # opacities = (opacities - np.min(opacities)) / (np.max(opacities) - np.min(opacities))
    scatter = plot_points(xyz, opacities, sizes)
    # from pdb import set_trace; set_trace()
    fig = go.Figure(data=[scatter])
    fig.update_layout(
        scene=dict(
            xaxis=dict(visible=False),
            yaxis=dict(visible=False),
            zaxis=dict(visible=False),
            aspectmode='data',
            bgcolor = 'rgb(20, 24, 54)'
        ),
        margin=dict(l=0, r=0, b=0, t=0),
    )

    # Show the figure
    
    py.write_html(fig, f'/home/qianxu/Project/Explore_LVM/pc.html')