import os,sys
import BasicUseFunc as basFunc
import numpy as np
import re
from scipy.spatial.transform import Rotation as R
from typing import Sequence,Tuple,List
try:
    import open3d as o3d
    o3dflag=True
except:
    o3dflag=False
    print('open3d not found in pip,you can try pip install open3d')
'''
pts:N,m(m可以是3也可以是6)
'''
def WriteAsc3DPoints(file,pts,split=','):
    fp=open(file,'w')
    for i in range(pts.shape[0]):
        p=pts[i].tolist()
        strp=split.join([str(m) for m in p])
        fp.write(strp+'\n')
    fp.close()
def WriteAsc3DPointsColor(file:str,pts:np.ndarray,split=','):
    assert(pts.shape[1]==6)
    fp=open(file,'w')
    for i in range(pts.shape[0]):
        p=pts[i,:3].tolist()
        c=pts[i,3:].astype(np.uint64).tolist()
        strp=split.join([str(m) for m in p])+','+','.join([str(m) for m in c])
        fp.write(strp+'\n')
    fp.close()
def WriteBin3DPoints(file,pts):
    pts.astype(np.float32).tofile(file)
def ReadAsc3DPoints(file):
    fp=open(file,'r')
    txts=fp.readlines()
    fp.close()
    arrs=[]
    for txt in txts:
        arr=[float(s) for s in re.findall('(?:-|)[0-9]+(?:\.?[0-9]+|)(?:e-?[0-9]+|)', txt)]
        if len(arr)<3:
            continue
        checkarr=re.findall('[a-zA-z]',txt)
        flag=True
        for a in checkarr:
            if len(a)>=2:
                flag=False
                break
        if not flag:continue
        arrs.append(arr)
    ms=0
    for i in range(len(arrs)):
        ms=max(ms,len(arrs[i]))
    for i in range(len(arrs)):
        for j in range(len(arrs[i]),ms):
            arrs[i].append(0)
    arrs=np.array(arrs,np.float32)
    return arrs
def ReadBin3DPoints(file,shape=3):
    data=np.fromfile(file,np.float32)
    data=data.reshape([-1,shape])
    return data
def FitPlane(xyz):
    centers=xyz.mean(0)
    xyzs=xyz-centers[None,...]
    txyzs=np.transpose(xyzs,[1,0])
    mat=txyzs@xyzs
    W,V=np.linalg.eig(mat)
    wdex=W.argmin()
    vec=V[:,wdex]
    d=-np.dot(vec,centers)
    vec=np.concatenate([vec,[d]],0)
    return vec,centers
def GetDistancePts2Plane(xyz,vec):
    xyzs=np.concatenate([xyz,np.ones([xyz.shape[0],1],np.float32)],-1)
    a=np.linalg.norm(vec[:3])
    vecs=vec[:,None]
    b=np.abs(np.dot(xyzs,vecs)).squeeze()
    return b/a
def PerfectFitPlane(xyz,outdist=1.0):
    xyzs=xyz.copy()
    vec=None
    while(True):
        vec,centers=FitPlane(xyzs)
        dist=GetDistancePts2Plane(xyzs,vec)
        msk=dist<outdist
        if np.logical_not(msk).sum()==0:
            break
        xyzs=xyzs[msk]
    return vec,centers
'''
xyzpts(N,3)float32
indexs(N,3)int
'''
def OutPutTriangles(xyzpts,indxs,filename):
    txts=[]
    txts.append(f'# Data\ng data\n# Number of geometric vertices: {len(xyzpts)}')
    for i in range(xyzpts.shape[0]):
        xyz=xyzpts[i].tolist()
        xyz=xyz[:3]
        txts.append('v '+' '.join([str(xyz[i]) for i in range(len(xyz))]))
    indexs=indxs.copy()
    txts.append(f'# Number of texture vertex coordinates: 0\ng data Triangles_0\n# Number of triangles: {len(indexs)}')
    for i in range(indexs.shape[0]):
        index=indexs[i]
        index+=1
        txts.append('f '+' '.join([str(dex) for dex in index]))
    for i,txt in enumerate(txts):
        txts[i]+='\n'
    txts.append('# end of file')
    fp=open(filename,'w')
    fp.writelines(txts)
    fp.close()
def GetAngleFromTwoVec(vec0,vec1):
    nvec0=np.linalg.norm(vec0,axis=-1)
    nvec0=vec0/nvec0
    nvec1=np.linalg.norm(vec1,axis=-1)
    nvec1=vec1/nvec1
    res=np.dot(nvec0,nvec1)
    return np.arccos(res)

'''
center:[3]
vec:[3/4]
'''
def OutPlane(center,vec,norm=5.0):
    cvec=np.cross(vec[:3],np.array([0,0,1],np.float32))
    cnorm=np.linalg.norm(cvec)
    cvec/=cnorm
    w=GetAngleFromTwoVec(vec[:3],np.array([0,0,1],np.float32))
    w/=2.0
    w=-w
    wcos=np.cos(w)
    wsin=np.sin(w)    
    cvec*=wsin
    cvec=np.concatenate([cvec,[wcos]],-1)
    cvec=cvec.tolist()
    mat=R.from_quat(cvec)    
    rotation_matrix = mat.as_matrix()
    #testvec=rotation_matrix@(np.array([0,0,1],np.float32)[...,None])
    oripts=[[norm,0,0],[-norm,0,0],[0,norm,0],[0,-norm,0]]
    oripts=np.array(oripts,np.float32)
    oridex=[[1,3,0],[0,2,1]]
    index=np.array(oridex,np.int32)
    newpts=rotation_matrix[None,...]@oripts[...,None]
    newpts=newpts.squeeze()
    newpts+=center[None,...]
    return newpts,index
def GetLineInterSectionPlane(line_points, line_vec, plane_normal):
    """获取多条直线和一个平面的交点，输出(n, 3),需要排除平行和重合的情况

    Args:
        line_points (np.array, float): 直线上的点
        line_vec (np.array, float): 直线的方向向量
        plane_normal (np.array, float): 平面法向量
        plane_d (np.array, float): 平面的距离参数
    """    
    line_points = line_points.reshape(-1, 3)
    line_vec = line_vec.reshape(-1, 3)
    line_vec_norm=np.linalg.norm(line_vec,axis=1)
    line_vec/=line_vec_norm[...,None]
    plane_normal_norm=np.linalg.norm(plane_normal[:3])
    plane_normal/=plane_normal_norm
    param = -(plane_normal[3] + line_points.dot(plane_normal[:3])) / line_vec.dot(plane_normal[:3])  # (n,3)
    line_out = line_points + param[...,None] * line_vec     # (n,3)
    return line_out
#camMatrix 4x4
def OutCameraPosition(camMatrix:np.ndarray,focus:float=1,radius:float=1)->Tuple[np.ndarray,np.ndarray]:
    corners = np.array([[-1.0, -1.0],[1.0, -1.0], [1.0,1.0], [-1.0, 1.0]],np.float32)
    corners*=radius
    ones=np.ones([corners.shape[0],2],np.float32)
    ones[:,0]*=focus
    corners=np.concatenate([corners,ones],-1)
    i = [0, 0, 0, 0]
    j = [1, 2, 3, 4]
    k = [2, 3, 4, 1]
    triangles = np.vstack((i, j, k)).T
    vertices = np.concatenate(([[0,0,0,1.0]], corners))
    vertices=np.transpose(vertices,[1,0])
    vertices=camMatrix@vertices
    vertices=np.transpose(vertices,[1,0])
    return triangles,vertices[...,:3]
#vrtris=[[vertexs,triangles]]
def FuseAllTriVertex(vertexs:List[np.ndarray],triangles:List[np.ndarray])->Tuple[np.ndarray,np.ndarray]:
    vertex=[]
    triangle=[]
    start=0
    for i in range(len(vertexs)):
        vtx,tri=vertexs[i],triangles[i]
        tri+=start
        vertex.append(vtx)
        triangle.append(tri)
        start+=vtx.shape[0]
    vertexs=np.concatenate(vertex,0)
    triangles=np.concatenate(triangle,0)
    return triangles,vertexs
'''
camMatrix:N,4,4/N,3,4
fileDir:如果是文件夹就一个camera一个文件否则是多个camera一个文件
'''
def OutPutCameraPositionFile(fileDir:str,camMatrixs:np.ndarray,
                             useEmpthDir=False,**kwargs):
    if useEmpthDir:
        basFunc.MakeEmptyDir(fileDir)
    if os.path.isdir(fileDir):
        fuseflag=False
    else:
        fuseflag=True
    assert(len(camMatrixs.shape)==3 and camMatrixs.shape[2]==4)
    if camMatrixs.shape[1]==3:
        ones=np.array([0,0,0,1],np.float32)
        ones=np.tile(ones[None,None],[camMatrixs.shape[0],1,1])
        camMatrixs=np.concatenate([camMatrixs,ones],1)
    tris,vers=[],[]
    for i in range(camMatrixs.shape[0]):
        camMatrix=camMatrixs[i]
        tri,ver=OutCameraPosition(camMatrix,**kwargs)
        tris.append(tri)
        vers.append(ver)
    if not fuseflag:
        for i in range(len(tris)):
            tri,ver=tris[i],vers[i]
            file=os.path.join(fileDir,str(i)+'.obj')
            OutPutTriangles(ver,tri,file)
    else:
        ntris,nvers=FuseAllTriVertex(vers,tris)
        OutPutTriangles(nvers,ntris,fileDir)
def open3dmethod(func):
    def newfunc(*args,useNpy=False,**kwargs):
        global o3dflag
        if o3dflag:
            return func(*args,**kwargs)
        else:
            raise ModuleNotFoundError('not found open3d')
    return newfunc
@open3dmethod
def open3d_readpoints(file:str)->np.ndarray:
    ptcloud=o3d.io.read_point_cloud(file)
    return np.asarray(ptcloud.points).astype(np.float32)
@open3dmethod
def open3d_readmeshes(file:str)->Tuple[np.ndarray,np.ndarray]:
    mesh=o3d.io.read_triangle_mesh(file)
    pts=np.asarray(mesh.vertices,np.float32)
    index=np.asarray(mesh.triangles,np.int32)
    return pts,index
@open3dmethod
def open3d_writepoints(xyzpts:np.ndarray,file:str,write_ascii=True,**kwargs)->None:
    dxyzpts=xyzpts.astype(np.float64)
    points_o3d = o3d.utility.Vector3dVector(dxyzpts)
    ptcloud = o3d.geometry.PointCloud()
    ptcloud.points=points_o3d
    o3d.io.write_point_cloud(file,ptcloud,write_ascii=write_ascii,**kwargs)
    return
@open3dmethod
def open3d_writemeshes(xyzpts:np.ndarray,indexs:np.ndarray,file:str,**kwargs)->None:
    dxyzpts=xyzpts.astype(np.float64)
    indexes=indexs.astype(np.int32)
    points_o3d = o3d.utility.Vector3dVector(dxyzpts)
    index_o3d=o3d.utility.Vector3iVector(indexes)
    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(points_o3d)
    mesh.triangles = o3d.utility.Vector3iVector(index_o3d)
    o3d.io.write_triangle_mesh(file, mesh,**kwargs)
    return 
@open3dmethod
def open3d_numpy3d2PointCloud(xyzpts:np.ndarray)->o3d.geometry.PointCloud:
    dxyzpts=xyzpts.astype(np.float64)
    points_o3d = o3d.utility.Vector3dVector(dxyzpts)
    ptcloud = o3d.geometry.PointCloud()
    ptcloud.points=points_o3d
    return ptcloud
@open3dmethod
def open3d_numpy3d3i2TriangleMesh(xyzpts:np.ndarray,indexs:np.ndarray)->o3d.geometry.TriangleMesh:
    dxyzpts=xyzpts.astype(np.float64)
    indexes=indexs.astype(np.int32)
    points_o3d = o3d.utility.Vector3dVector(dxyzpts)
    index_o3d=o3d.utility.Vector3iVector(indexes)
    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(points_o3d)
    mesh.triangles = o3d.utility.Vector3iVector(index_o3d)
    return mesh
@open3dmethod
def open3d_PointCloud2numpy3d(ptcloud:o3d.geometry.PointCloud)->np.ndarray:
    return np.asarray(ptcloud.points).astype(np.float32)
@open3dmethod
def open3d_TriangleMesh2numpy3d3i(triMesh:o3d.geometry.TriangleMesh)->Tuple[np.ndarray,np.ndarray]:
    pts=np.asarray(triMesh.vertices,np.float32)
    index=np.asarray(triMesh.triangles,np.int32)
    return pts,index
@open3dmethod
def open3d_center_rotate_extend_boundbox(xyz:list,rotate:list,extend:list
                                         ,color:list=[0,0,0])\
    ->o3d.geometry.OrientedBoundingBox:
    r=o3d.geometry.get_rotation_matrix_from_xyz(tuple(rotate))
    obb = o3d.geometry.OrientedBoundingBox(xyz, r, extend)
    obb.color=color
    return obb
@open3dmethod
def open3d_centers_extends_rots_boundboxes(xyzs:np.ndarray,extends:np.ndarray,
                                           rots:np.ndarray,color=None)->\
    List[o3d.geometry.OrientedBoundingBox]:
    assert(len(xyzs.shape)==2)
    assert(len(extends.shape)==2)
    assert(len(rots.shape)==2)
    assert(xyzs.shape[0]==extends.shape[0])
    assert(xyzs.shape[0]==rots.shape[0])
    if color is not None:
        assert(type(color) is np.ndarray)
        assert(len(color.shape)==2)
        assert(color.shape[0]==xyzs.shape[0])
    else:
        color=np.zeros_like(xyzs).astype(np.int32)
    boxes=[]
    for i in range(xyzs.shape[0]):
        obb=open3d_center_rotate_extend_boundbox(xyzs[i].tolist(),rots[i].tolist()
                                                 ,extends[i].tolist(),color[i].tolist())
        boxes.append(obb)
    return boxes
@open3dmethod
def open3d_centers_extends_yaws_boundboxes(xyzs:np.ndarray,extends:np.ndarray,
                                           yaws:np.ndarray,color=None)->\
    List[o3d.geometry.OrientedBoundingBox]:
    assert(len(yaws.shape)==1)
    assert(yaws.shape[0]==xyzs.shape[0])
    rots=np.zeros_like(xyzs)
    rots[...,2]=yaws
    return open3d_centers_extends_rots_boundboxes(xyzs,extends,rots,color)
@open3dmethod
def open3d_drawgeometries(geobjlst:list,size:float=-1.0):
    if size>0:
        frame=o3d.geometry.TriangleMesh.create_coordinate_frame(size=1.0, origin=[0, 0, 0])
        geobjlst.append(frame)
    o3d.visualization.draw_geometries(geobjlst)
'''
camMatrix:N,4,4/N,3,4
'''
def OutCameraPositionInv(camMatrix:np.ndarray,focus:float=1,radius:float=1)->Tuple[np.ndarray,np.ndarray]:
    corners = np.array([[-1.0, -1.0],[1.0, -1.0], [1.0,1.0], [-1.0, 1.0]],np.float32)
    corners*=radius
    ones=np.ones([corners.shape[0],2],np.float32)
    ones[:,0]*=focus
    corners=np.concatenate([corners,ones],-1)
    i = [0, 0, 0, 0]
    j = [1, 2, 3, 4]
    k = [2, 3, 4, 1]
    triangles = np.vstack((k,j,i)).T
    vertices = np.concatenate(([[0,0,0,1.0]], corners))
    vertices=np.transpose(vertices,[1,0])
    vertices=camMatrix@vertices
    vertices=np.transpose(vertices,[1,0])
    return triangles,vertices[...,:3]
def open3d_outcamerageo(camMatrixs:np.ndarray,fuseflag:bool=True,**kwargs):
    if camMatrixs.shape[1]==3:
        ones=np.array([0,0,0,1],np.float32)
        ones=np.tile(ones[None,None],[camMatrixs.shape[0],1,1])
        camMatrixs=np.concatenate([camMatrixs,ones],1)
    tris,vers=[],[]
    for i in range(camMatrixs.shape[0]):
        camMatrix=camMatrixs[i]
        triI,ver=OutCameraPositionInv(camMatrix,**kwargs)
        tri,ver=OutCameraPosition(camMatrix,**kwargs)
        tri=np.concatenate([tri,triI],0)
        tris.append(tri)
        vers.append(ver)
    if not fuseflag:
        geos=[]
        for i in range(len(tris)):
            tri,ver=tris[i],vers[i]
            trimesh=open3d_numpy3d3i2TriangleMesh(ver,tri)
            geos.append(trimesh)
        return geos
    else:
        ntris,nvers=FuseAllTriVertex(vers,tris)
        trimesh=open3d_numpy3d3i2TriangleMesh(nvers,ntris)
        return trimesh