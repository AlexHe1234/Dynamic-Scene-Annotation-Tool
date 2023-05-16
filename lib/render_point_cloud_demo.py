from .general_import import *
import render_point_cloud as rpc


R = 10
focal = 800.
frame_per_round = 100

edge = R / 5
pts = np.array([[1, 1, 1],
                  [1, 1, -1],
                  [1, -1, 1],
                  [1, -1, -1],
                  [-1, 1, 1],
                  [-1, 1, -1],
                  [-1, -1, 1],
                  [-1, -1, -1]], dtype=np.float32)
pts *= edge

r = np.array([[1., -1., 1.],
              [1., 0., -2.],
              [1., 1., 1.]])
r[:, 0] /= np.linalg.norm(r[:, 0])
r[:, 1] /= np.linalg.norm(r[:, 1])
r[:, 2] /= np.linalg.norm(r[:, 2])
pts = pts.dot(r.T)

delta = np.pi * 2 / frame_per_round
theta = 0

# todo
k = np.array([[focal, 0, 400],
              [0, focal, 400],
              [0, 0, 1]])

while True:
    t_ = R * np.array([np.cos(theta), np.sin(theta), 0])
    r_ = np.array([[-np.sin(theta), 0, -np.cos(theta)],
                   [np.cos(theta), 0, -np.sin(theta)],
                   [0, -1, 0]])
    r = r_.T
    t = -r.dot(t_)
    img = rpc.render_point_cloud(k, r, t, pts, 800, 800, 5)
    cv2.imshow('demo', img)
    cv2.waitKey(int(1000 / 60.))
    theta += delta
    theta %= np.pi * 2
