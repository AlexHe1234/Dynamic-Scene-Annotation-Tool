import numpy as np


def plot(pts: np.ndarray, use_plotly=True, color='blue', size=5):
    """quickly plot points using either plotly or pyplot

    Args:
        pts (np.ndarray): numpy array of 3d points, shape [SIZE, 3]
        use_plotly (bool, optional): True for plotly, False for matplotlib. Defaults to True.
        color (str, optional): color for the points, choose from 'blue', 'yellow', 'red', 'green' etc.
        size (int, optional): point size.
    """
    pts = pts.reshape(-1, 3)
    if use_plotly:
        import plotly.graph_objects as go

        # Create a trace for the 3D scatter plot
        trace = go.Scatter3d(x=pts[:, 0], y=pts[:, 1], z=pts[:, 2], mode='markers', marker=dict(size=size, color=color))

        # Create a layout for the plot
        layout = go.Layout(scene=dict(
            xaxis=dict(title='X Axis'),
            yaxis=dict(title='Y Axis'),
            zaxis=dict(title='Z Axis')
        ))

        # Combine the trace and layout and create the figure
        fig = go.Figure(data=[trace], layout=layout)

        # Show the plot
        fig.show()
    else:
        import matplotlib.pyplot as plt
        
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2], c=color, s=size)
        plt.show()
        
        
if __name__ == '__main__':
    example = np.random.randn(1000, 3)
    plot(example)
    