import os,sys
import cvSDK.BasicUseFunc as basFunc
import tkinter as tk
from tkinter import filedialog
import cv2
from direct.showbase.ShowBase import ShowBase
from panda3d.core import Material, TransparencyAttrib
from panda3d.core import Filename
import numpy as np
from panda3d.core import DirectionalLight, PointLight, Spotlight, AmbientLight
from panda3d.core import NodePath, Vec4, Filename
from panda3d.core import PNMImage, Texture, NodePath, CardMaker, WindowProperties,loadPrcFileData, NodePath, LVecBase4f, LPoint3f,LMatrix4f
from panda3d.core import GeomVertexFormat, GeomVertexData, Geom, GeomVertexWriter, GeomLines,GeomNode
import cvSDK.BasicPicDeal as basPic
import cvSDK.DataIO as dio
import math
from math import *
def get_R(x,y,z):
    ''' Get rotation matrix from three rotation angles (radians). right-handed.
    Args:
        angles: [3,]. x, y, z angles
    Returns:
        R: [3, 3]. rotation matrix.
    '''
    # x
    Rx = np.array([[1, 0, 0],
                   [0, np.cos(x), -np.sin(x)],
                   [0, np.sin(x), np.cos(x)]])
    # y
    Ry = np.array([[np.cos(y), 0, np.sin(y)],
                   [0, 1, 0],
                   [-np.sin(y), 0, np.cos(y)]])
    # z
    Rz = np.array([[np.cos(z), -np.sin(z), 0],
                   [np.sin(z), np.cos(z), 0],
                   [0, 0, 1]])

    R = Rz.dot(Ry.dot(Rx))
    return R


folder_selected = filedialog.askdirectory()
if(folder_selected is None):
    exit(0)
files=basFunc.getdatas(folder_selected,"*.jpg")
files.extend(basFunc.getdatas(folder_selected,"*.jpeg"))
files.extend(basFunc.getdatas(folder_selected,"*.png"))
if(len(files)==0):
    exit(0)
print("use folder",folder_selected,"files:",len(files))
idx=0
curidx=-1
img=None
def compute_euler_angles_from_rotation_matricesnpy(rotation_matrices):
    batch = rotation_matrices.shape[0]
    R = rotation_matrices
    sy = np.sqrt(R[:,0,0]*R[:,0,0]+R[:,1,0]*R[:,1,0])
    singular = sy<1e-6
    x = np.arctan2(R[:,2,1], R[:,2,2])
    y = np.arctan2(-R[:,2,0], sy)
    z = np.arctan2(R[:,1,0],R[:,0,0])
    
    xs = np.arctan2(-R[:,1,2], R[:,1,1])
    ys = np.arctan2(-R[:,2,0], sy)
    zs = R[:,1,0]*0
    out_euler=np.zeros([batch,3],np.float32)
    out_euler[:,0] = x*(1-singular)+xs*singular
    out_euler[:,1] = y*(1-singular)+ys*singular
    out_euler[:,2] = z*(1-singular)+zs*singular
        
    return out_euler
def draw_axis(img, yaw, pitch, roll, tdx=None, tdy=None, size = 100):

    pitch = pitch * np.pi / 180
    yaw = -(yaw * np.pi / 180)
    roll = roll * np.pi / 180

    if tdx != None and tdy != None:
        tdx = tdx
        tdy = tdy
    else:
        height, width = img.shape[:2]
        tdx = width / 2
        tdy = height / 2

    # X-Axis pointing to right. drawn in red
    x1 = size * (cos(yaw) * cos(roll)) + tdx
    y1 = size * (cos(pitch) * sin(roll) + cos(roll) * sin(pitch) * sin(yaw)) + tdy

    # Y-Axis | drawn in green
    #        v
    x2 = size * (-cos(yaw) * sin(roll)) + tdx
    y2 = size * (cos(pitch) * cos(roll) - sin(pitch) * sin(yaw) * sin(roll)) + tdy

    # Z-Axis (out of the screen) drawn in blue
    x3 = size * (sin(yaw)) + tdx
    y3 = size * (-cos(yaw) * sin(pitch)) + tdy

    cv2.line(img, (int(tdx), int(tdy)), (int(x1),int(y1)),(0,0,255),4)
    cv2.line(img, (int(tdx), int(tdy)), (int(x2),int(y2)),(0,255,0),4)
    cv2.line(img, (int(tdx), int(tdy)), (int(x3),int(y3)),(255,0,0),4)

    return img

class MyApp(ShowBase):

    def __init__(self):
        ShowBase.__init__(self)
        # 加载 .obj 文件
        model_path = "01.obj"
        self.model = self.loader.loadModel(model_path)
        min_point, max_point = self.model.getTightBounds()

        # 计算中心点
        center = (min_point + max_point) / 2
        center=np.array([center[0],center[1],center[2]],np.float32)
        #center=-center
        self.model.reparentTo(self.render)
        dlight = DirectionalLight('dlight')
        dlight.setColor(Vec4(0.8, 0.8, 0.8, 1))
        dlnp = self.render.attachNewNode(dlight)
        dlnp.setHpr(0, -135, 0)
        self.render.setLight(dlnp)
        # 设置模型的缩放
        self.model.setScale(1.0)
        props = WindowProperties()
        props.setSize(600, 600)
        # 应用窗口属性
        self.win.requestProperties(props)
        # 设置背景颜色
        self.setBackgroundColor(0.5, 0.5, 0.5)

        # 添加一个简单的摄像机控制
        self.disableMouse()
        self.camera.setPos(0, 0, 1000)
        self.camera.lookAt(0, 0, 0)
        self.idx=0
        self.curidx=-1
        self.img=None
        # 添加一个任务来更新模型的位置
        self.taskMgr.add(self.update_model, "update_model_task")
        self.taskMgr.add(self.capture_frame, 'capture_frame')
        self.accept("a", self.a)
        self.accept("d", self.d)
        self.accept('mouse1', self.on_mouse_down)
        self.accept('mouse1-up', self.on_mouse_up)
        self.accept('mouse3', self.on_mouse2_down)
        self.accept('mouse3-up', self.on_mouse2_up)
        self.mat=np.eye(4,dtype=np.float32)
        self.orimat=self.mat
        self.orimat[:3,:3]=get_R(0,math.pi,math.pi)
        self.orimat[3,:3]=center
        self.is_dragging = False
        self.sampleframe=1
        self.infercnt=0
        self.drag_start_pos=None
        self.Create3DGui()
    def Create3DGui(self):
        # 创建圆形几何形状
        num_segments = 128  # 圆形的细分段数
        self.radius = 0.8  # 圆形的半径
        radius=self.radius
        format = GeomVertexFormat.getV3c4()
        vdata = GeomVertexData('circle', format, Geom.UHStatic)

        vertex = GeomVertexWriter(vdata, 'vertex')
        color = GeomVertexWriter(vdata, 'color')

        for i in range(num_segments + 1):
            angle = (i / num_segments) * 2 * pi
            x = radius * cos(angle)
            y = radius * sin(angle)
            vertex.addData3(x, 0, y)
            color.addData4(1, 1, 1, 1)  # 红色

        prim = GeomLines(Geom.UHStatic)
        for i in range(num_segments):
            prim.addVertices(i, (i + 1) % num_segments)

        geom = Geom(vdata)
        geom.addPrimitive(prim)

        node = GeomNode('gnode')
        node.addGeom(geom)

        # 创建 NodePath 并添加到场景中
        circle = self.render2d.attachNewNode(node)
        circle.reparentTo(self.aspect2d)
        circle.setPos(0, 0, 0)  # 屏幕中心
    def RotateMat(self):
        allmat=self.orimat@self.mat
        lmat=LMatrix4f(*allmat.flatten().tolist())
        self.model.setMat(lmat)
    def on_mouse2_down(self):
        self.mat=np.eye(4,dtype=np.float32)
    def on_mouse2_up(self):
        return
    def on_mouse_down(self):
        if self.mouseWatcherNode.hasMouse():
            self.is_dragging = True

    def on_mouse_up(self):
        self.is_dragging = False
        self.drag_start_pos = None
        global files
        file=files[self.idx]
        d,name,ftr=basFunc.GetfileDirNamefilter(file)
        pklfile=os.path.join(d,name+".pkl")
        out=compute_euler_angles_from_rotation_matricesnpy(self.mat[None,...])[0]
        pitch,yaw,roll=out.tolist()
        dio.SaveVariableToPKL(pklfile,[[],out.tolist()])
        dct={}
        dct['euler_angle']=[pitch,yaw,roll]
        dct['mat']=self.mat.tolist()
        jsonfile=os.path.join(d,name+".json")
        dio.writejsondictFormatFile(dct,jsonfile)
        return
    def a(self):
        self.idx-=1
        self.idx=max(self.idx,0)

    def d(self):
        self.idx+=1
        self.idx=min(self.idx,len(files)-1)

    def update_model(self, task):
        global files
        if(self.curidx!=self.idx):
            self.img=cv2.imread(files[self.idx])
            print(files[self.idx],f'{self.idx+1}/len(files)')
            self.img=basPic.GenerateExpandImageData(self.img,400,400)
            self.debugimg=self.img.copy()
            self.curidx=self.idx
            self.mat=np.eye(4,dtype=np.float32)
            d,name,ftr=basFunc.GetfileDirNamefilter(files[self.idx])
            pklfile=os.path.join(d,name+".pkl")
            if(os.path.exists(pklfile)):
                [outbox,outrot]=dio.LoadVariablefromPKL(pklfile)
                draw_axis(self.debugimg,outrot[1]*180/np.pi,outrot[0]*180/np.pi,outrot[2]*180/np.pi)
                rotmat=get_R(*outrot)
                self.mat[:3,:3]=rotmat
        if self.mouseWatcherNode.hasMouse() and self.is_dragging:
            mouse_pos = self.mouseWatcherNode.getMouse()
            posmouse=[mouse_pos[0],mouse_pos[1]]
            if self.drag_start_pos is not None:
                delta_x = posmouse[0] - self.drag_start_pos[0]
                delta_y = posmouse[1] - self.drag_start_pos[1]
                delta_z=0
                radius=math.sqrt(posmouse[0]*posmouse[0]+posmouse[1]*posmouse[1])
                if(radius>self.radius):
                    delta_x=delta_y=0
                    orinml=math.sqrt(self.drag_start_pos[0]*self.drag_start_pos[0]+self.drag_start_pos[1]*self.drag_start_pos[1])
                    nwnml=math.sqrt(posmouse[0]*posmouse[0]+posmouse[1]*posmouse[1])
                    ori=math.atan2(self.drag_start_pos[1]/orinml,self.drag_start_pos[0]/orinml)
                    nw=math.atan2(posmouse[1]/nwnml,posmouse[0]/nwnml)
                    delta_z=nw-ori
                delta_x*=-1
                delta_y*=1
                delta_z*=-1
                rotmat=np.eye(4,dtype=np.float32)
                rotmat[:3,:3]=get_R(delta_y,delta_x,delta_z)
                self.mat=self.mat@rotmat
                if self.infercnt%self.sampleframe==0:
                    self.drag_start_pos = posmouse
                    self.infercnt=0
                self.infercnt+=1
            else:
                self.drag_start_pos = posmouse
                self.infercnt=0
        self.RotateMat()    
        out=compute_euler_angles_from_rotation_matricesnpy(self.mat[None,...])[0]
        pitch,yaw,roll=out.tolist()
        self.debugimg=self.img.copy()
        draw_axis(self.debugimg,yaw*180/np.pi,pitch*180/np.pi,roll*180/np.pi)    
        return task.cont  
    def capture_frame(self, task):
        # 捕获渲染结果
        # 使用OpenCV显示图像
        allimg=np.concatenate([self.img,self.debugimg],0)
        cv2.imshow('Image', allimg)
        key=cv2.waitKey(1) & 0xFF
        if key==27:
            exit(0)
        return task.cont

app = MyApp()
app.run()



