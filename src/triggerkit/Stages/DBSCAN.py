import numpy as np
from sklearn.cluster import DBSCAN as SklearnDBSCAN
from ctapipe.instrument import CameraGeometry
from astropy import units as u


HEXAGON_SIZE_CM = 3.71236 
HORIZ_DISTANCE_CM = HEXAGON_SIZE_CM * np.sqrt(3)  # center-to-center horizontal distance
VERTICAL_DISTANCE_CM = 1.5 * HEXAGON_SIZE_CM  # center-to-center vertical distance

def tdscan_to_dbscan_params(eps_xy, eps_t, tolerance=1e-3):
    if( eps_xy == 0 ):
        dbscan_eps = HORIZ_DISTANCE_CM
    elif( eps_xy == 1 ):
        dbscan_eps = eps_xy * HEXAGON_SIZE_CM * 3
    elif( eps_xy == 2 ):
        dbscan_eps = eps_xy * HEXAGON_SIZE_CM * 5
    elif( eps_xy == 3 ):
        dbscan_eps = eps_xy * HEXAGON_SIZE_CM * 6
    else:
        raise ValueError(f"Invalid eps_xy value: {eps_xy}")
    z_spacing_cm = np.sqrt(dbscan_eps**2 - (HORIZ_DISTANCE_CM)**2) / eps_t
    dbscan_eps = dbscan_eps / 100.0  # convert to meters
    z_spacing_cm = z_spacing_cm / 100.0  # convert to meters
    return dbscan_eps-tolerance, z_spacing_cm-tolerance*2

class DBSCAN:
    def __init__(self, eps=0.11, min_points=3, time_norm=0.09):
        self.eps = eps
        self.min_points = np.uint32(min_points)
        self.time_norm = time_norm
        self.output_geometry = None
        self.input_geometry = None
        self.output_unit = "-"
    
    def stage_name(self):
        return f"dbscan{self.eps:.2f}_{self.min_points}_{self.time_norm:.2f}"
    
    def stage_type(self):
        return "dbscan"
    
    def get_params(self):
        return {
            'eps': self.eps,
            'min_points': self.min_points,
            'time_norm': self.time_norm
        }

    def get_stages(self):
        return (self.stage_type(), self.get_params())

    def execute(self, wf, **kwargs):
        # get the x,y positions of all triggered pixels (wf>0) with self.input_geometry.pix_x and pix_y
        X = np.argwhere(wf > 0)
        pix_idx = X[:, 0]
        t_idx   = X[:, 1]
        points_x = self.input_geometry.pix_x[pix_idx]
        points_y = self.input_geometry.pix_y[pix_idx]
        # convert points_x and points_y to dimensionless
        points_x = points_x.to_value(u.m)
        points_y = points_y.to_value(u.m)
        points_t = t_idx * self.time_norm  # normalize time
        X = np.column_stack((points_x, points_y, points_t))  # stack x,y,time
        # print the number of points to be clustered
        if X.shape[0] == 0:
            return np.zeros_like(wf)
        dbscan = SklearnDBSCAN( eps = self.eps, min_samples = self.min_points)
        clusters = dbscan.fit_predict(X)
        wf_out = np.zeros_like(wf)
        valid  = clusters != -1
        wf_out[pix_idx[valid], t_idx[valid]] = 1
        return wf_out
    
    def compile(self, input_geometry: CameraGeometry):
        self.input_geometry = input_geometry
        # for now output the same geometry as input
        self.output_geometry = self.input_geometry
        return self.output_geometry

    def printInfo(self):
        print(f"DBSCAN Stage: eps={self.eps}, min_points={self.min_points}, time_norm={self.time_norm}")
        print(f"Input Geometry: {len(self.input_geometry.pix_x)} pixels")
        print(f"Output Geometry: {len(self.output_geometry.pix_x)} pixels")
