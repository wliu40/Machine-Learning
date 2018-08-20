# -*- coding: utf-8 -*-
"""
Created on Fri Aug 17 21:49:53 2018

@author: Arvin
"""

#%%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt



lon_min = 0
lon_max = 5
lat_min = 0
lat_max = 5
#x = np.linspace(x_min, x_max, nx)  # generate 6 points, in range[0,5]: [0. 1. 2. 3. 4. 5.]
#y = np.linspace(y_min, y_max, ny)

#nlon = 6
#nlat = 6
#unit_lon = (lon_max-lon_min)/(nlon-1)
#unit_lat = (lat_max-lat_min)/(nlat-1)

unit_lon = 1
unit_lat = 1

lon_pos = np.array(range(lon_min, lon_max, unit_lon)) + unit_lon/2.0
lat_pos = np.array(range(lat_min, lat_max, unit_lat)) + unit_lat/2.0

num_lon = len(lon_pos)
num_lat = len(lat_pos)

lons, lats = np.meshgrid(lon_pos, lat_pos)
#lons, lats = np.mgrid[lon_pos.tolist(), lat_pos.tolist()]
centers = np.c_[lons.ravel(), lats.ravel()]
centers  # 25 elements

class Point():
    def __init__(self, lon, lat):
        self.lon = lon
        self.lat = lat
    def __str__(self):
        return '({0}, {1})'.format(self.lon, self.lat)
    def haversine_dist(self, other):
        from math import radians, cos, sin, asin, sqrt
        lon1, lat1, lon2, lat2 = map(radians, [self.lon, self.lat, 
                                               other.lon, other.lat])
        dlon = lon2 - lon1 
        dlat = lat2 - lat1 
        a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
        c = 2 * asin(sqrt(a)) 
        r = 6371 # Radius of earth in kilometers.
        return c * r
    def euclidean_dist(self, other):
        lon1, lat1, lon2, lat2 = self.lon, self.lat, other.lon, other.lat
        return np.sqrt((lon1 - lat1)**2 + (lon2 - lat2)**2)
        

class Cells():
    
    #from Point import Point
    class Cell():
        def __init__(self, center, idx, unit_lon, unit_lat):
            #print (Point(center[0], center[1]))
            self.center = Point(center[0], center[1])
            self.idx = idx
            self.left = self.center.lon - unit_lon/2.0
            self.right = self.center.lon + unit_lon/2.0
            self.up = self.center.lat + unit_lat/2.0
            self.bottom = self.center.lat - unit_lat/2.0
            
        def __str__(self):
            return 'center:{0}, index:{1}'.format(str(self.center), self.idx)
        
    def __init__(self, lon_min, lon_max, unit_lon, lat_min, lat_max, unit_lat):
        import numpy as np
        self.lon_min = lon_min
        self.lon_max = lon_max
        self.unit_lon = unit_lon
        self.lat_min = lat_min
        self.lat_max = lat_max
        self.unit_lat = unit_lat        
        lon_pos = np.array(range(lon_min, lon_max, unit_lon)) + unit_lon/2.0
        lat_pos = np.array(range(lat_min, lat_max, unit_lat)) + unit_lat/2.0
        self.num_lon = len(lon_pos)
        self.num_lat = len(lat_pos)
        lons, lats = np.meshgrid(lon_pos, lat_pos)
        #lons, lats = np.mgrid[lon_pos.tolist(), lat_pos.tolist()]
        centers = np.c_[lons.ravel(), lats.ravel()]
        self.centers = centers
        self.cells = []
        for i,c in enumerate(centers):  # each cell has its index and center
            self.cells.append(Cell(c, i, unit_lon, unit_lat))
            # how to get cell with index of i? -> cells[i]
        self.tree = self.build_kdtree(centers)
    
    def get_cell_from_point(self, point):  # given a point, return which cell it is in   
        _, idx = self.tree.query((point.lon, point.lat), k=1)
        current_cell = self.cells[idx]
        return current_cell
    
    def get_cell_from_index(self, index):  # given a index, return the cell object
        return self.cells[index]

    def build_kdtree(self, centers):        
        from scipy.spatial import cKDTree
        tree = cKDTree(centers)
        return tree
        
    def check_cell_boundary(self, point):
        res = self.get_cell_from_point(point)
        cond = (res.left < point.lon < res.right)
        return cond and (res.bottom < point.lat < res.up)
    
                
                
    def get_neighbor_cell(self, idx, direct):
        if direct == 'left' and self.cells[idx-1].right < self.cells[idx].right:
            return self.cells[idx-1]
        elif direct == 'right' and idx+1 < len(self.cells) and self.cells[idx+1].left > self.cells[idx].left:
            return self.cells[idx+1]
        elif direct == 'up' and idx+self.num_lon < len(self.cells) and self.cells[idx+self.num_lon].up > self.cells[idx].up:
            return self.cells[idx+self.num_lon]
        elif direct == 'bottom' and self.cells[idx-self.num_lon].up < self.cells[idx].up:
            return self.cells[idx-self.num_lon]
        return None
    
    def get_close_cells(self, point, threshold):
        cur_cell = self.get_cell_from_point(point)
        res = [cur_cell]
        if abs(point.lon - cur_cell.left) < threshold:
            res.append(self.get_neighbor_cell(cur_cell.idx, 'left'))
        if abs(point.lon - cur_cell.right) < threshold:
            res.append(self.get_neighbor_cell(cur_cell.idx, 'right'))
        if abs(point.lat - cur_cell.up) < threshold:
            res.append(self.get_neighbor_cell(cur_cell.idx, 'up'))
        if abs(point.lat - cur_cell.bottom) < threshold:
            res.append(self.get_neighbor_cell(cur_cell.idx, 'bottom'))
        return list(filter(lambda x: x!=None), res)
    
        
#%%
        
    
#%%
    


## here should be all the (lon, lat) points
#rand_x = [np.random.uniform(x_min, x_max) for i in range(10)]
#rand_y = [np.random.uniform(x_min, x_max) for i in range(10)]
#rand_vars = zip(rand_x, rand_y)
#
#dd, ii = tree.query(list(rand_vars), k=1)
#
## ii is the index for each point
#print (centers[ii])

#

@pandas_udf('double', PandasUDFType.SCALAR)
def foo(lon_col, lat_col, k):
    return pd.Series(tree.query(list(zip(lon_col, lat_col), k=k)[1]))

df.withColumn('cell_idx', foo(df.lon, df.lat, 1))


def get_cell_label(centers, tree, lon, lat, threshold):
    _, idx = tree.query(point, k=1)
    current_cell = cells[idx]
    res = [current_cell]
    if abs(lon - current_cell.left) < threshold:
        res.append(get_neighbor_cell(idx, cells, ))
    
#    dists = [point.haversine_dist(Point(i[0], i[1])) for i in centers[idx]]
#    centers = list(zip(idx, dists))  # (center's idx, dist)
#    centers = sort(centers.items(), lambda x: x[1])  #rank from nearest to farthest 
#    if centers[0][1] < threshold:
#        return [centers[0][0]]
#    elif centers[0][1]
    


p = Point(1,2)
p1 = Point(3,4)




defects_df.withColumn('cell_idx', foo(defects_df.lon, defects_df.lat))


a = pd.Series([1,2,3])
print(list(zip(a, a)))

# give a point (x, y) -> cell_idx = 10 (e.g.), get the df in that cell
cell_points = df.filter[df['cell_idx'] == 10]


#%%
# assume one lon (经度) diff is 110*cos(b) km, 1 lat diff is 111 km (approx.)
from math import radians, cos, sin, asin, sqrt
def haversine(lon1, lat1, lon2, lat2):
    """
    Calculate the great circle distance between two points 
    on the earth (specified in decimal degrees)
    """
    # convert decimal degrees to radians 
    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])

    # haversine formula 
    dlon = lon2 - lon1 
    dlat = lat2 - lat1 
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * asin(sqrt(a)) 
    r = 6371 # Radius of earth in kilometers. Use 3956 for miles
    return c * r

lon_range = [-110, -70]
lat_range = [30, 45]

lons = np.random.uniform(-110, -70, size=50)
lats = np.random.uniform(30, 45, size=50)
points = list(zip(lons, lats))

h1s = []
h2s = []
for i in range(len(points)):
    for j in range(i+1, len(points)):
        p1, p2 = points[i], points[j]
        h1 = haversine(p1[0], p1[1], p2[0], p2[1]) 
        h2 = np.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)
        h1s.append(h1)  # haversine
        h2s.append(h2)  # euclidean
        

fig, ax = plt.subplots(figsize=(15,10))
ax.scatter(h2s, h1s)
ax.set_xlabel('haversine')
ax.set_ylabel('euclidian')
fig.show()
#%%
from sklearn import linear_model
regr = linear_model.LinearRegression()
# Train the model using the training sets

h2s = np.array(h2s).reshape(-1,1)
h1s = np.array(h1s).reshape(-1,1)
regr.fit(h1s, h2s)
#print (regr.coef_)
print ('formula: {} * x + {} = y'.format(regr.coef_[0][0], regr.intercept_[0]))
# 86.01705298284584 * x + 83.66801707234754 = y
# x is the haversine dist, y is the euclidean dist

# 100m = 0.1km =  (for euclidean)


#%%



#%%
a = pd.Series(['a','b','c'])
b = pd.Series([1,2,3])
c = (pd.DataFrame(np.c_[a,b], columns=['a','b']))
print (c)
c.head()
np.c_[a,b]
np.c_[['a', 'b'], [1,2]]
type(c.b.iloc[0])

#%%
print (np.c_[x.ravel(), y.ravel()][13])

xv, yv = np.meshgrid(x, y, sparse=False, indexing='ij')
fig, ax = plt.subplots(figsize=(22,10))
centers = []
for i in range(nx-1):
    for j in range(ny-1):
        centers.append((xv[i,j]+unit_x/2.0, yv[i,j]+unit_y/2.0))
        ax.scatter(xv[i,j]+unit_x/2.0, yv[i,j]+unit_y/2.0, marker='+', s=200)
plt.xlim(0,5)
plt.ylim(0,5)
#fig.show()

ax.scatter(x, y)
fig.show()

var = []
for i in range(40):
    ii = np.random.uniform(0, 5.0)
    jj = np.random.uniform(0, 5.0)
    var.append((ii, jj))
    ax.scatter(ii, jj)
        



dd, ii = tree.query([[2.1,2.9]], k=1)
#%%
print (dd) # the distances to the nearest point
print (ii) # the The index of the neighbors in the np.c_[x.ravel(), y.ravel()]
