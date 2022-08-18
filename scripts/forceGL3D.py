from OpenGL.GL import *
from OpenGL.GLUT import *
from OpenGL.GLU import *
import numpy as np
import math
#
class forceGL:
    def __init__(self, w, h):
        self.posB = []
        self.norB = []
        self.forces = []
        self.fov = 0
        self.aspect= 0
        self.near = 0
        self.far = 0
        self.cameraPos = 0
        self.target = 0
        self.up = 0
        self.w = w
        self.h = h
        self.max = 1
        self.data = []
        self.initGLUT()
        self.initGL()
        glutDisplayFunc(self.draw)
        
    def initGL(self):
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        glMatrixMode(GL_MODELVIEW)
        glLoadIdentity()
    def setForceLimits(self,min,max):
        '''in grams'''
        self.min = min
        self.max = max

    def getImgBuffer(self):
        data = glReadPixels(0, 0, self.w, self.h, GL_RGBA, GL_UNSIGNED_BYTE,outputType=0 ) 
        data = np.flipud(data).copy(order='C')
        return data
    
    def computeViewMatrix(self, cameraPos, target, up):
        self.cameraPos = cameraPos
        self.target = target
        self.up = up
    
    def computeProjectionMatrixFOV(self, fov, aspect, near, far):
        self.fov = fov
        self.aspect= aspect
        self.near = near
        self.far = far

    def view(self):
        glMatrixMode(GL_MODELVIEW)
        glLoadIdentity()
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()

    def initGLUT(self):
        glutInit()
        glutInitDisplayMode(GLUT_RGBA) 
        glutInitWindowSize(self.w,self.h)
        glutInitWindowPosition(0, 0)  
        wind = glutCreateWindow("OpenGL Force Image".encode('cp932')) 
        glutDisplayFunc(self.draw)
        glutReshapeFunc(self.reshape)
        glEnable(GL_BLEND) #OK
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA) #BEST
        glPixelStorei(GL_PACK_ALIGNMENT, 1) #temp
    
    def loopGLUT(self):
        glutMainLoop()

    def EventLoopGLUT(self):
        glutMainLoopEvent()

    def reshape(self, x, y):
        glutDisplayFunc(self.draw)
        glutPostRedisplay()    

    def setMesh(self, posB, norB, force):
        self.posB = posB
        self.norB = norB
        self.force = force

    def setForces(self, forces, pos, weight):
        self.forces = forces
        self.weight = weight
    #    self.norms = norms
        self.pos = pos

    def div(self, x,y):
        if y==0:
            return 0
        return x/y
    def circle(self, radius, force, pos,weight=[1,1,1]):
        x = pos[0]
        y = pos[1]
        glBegin(GL_TRIANGLE_FAN)
        #glColor3f(abs(force[2])/ self.max, abs(force[0])/ self.max, abs(force[1]) / self.max) #check here    
        #total = sum(force)
        
        R = force[0] * weight[0]
        G = force[1] * weight[1]
        B = force[2] * weight[2]
        if R > self.max:
            R = self.max
        if G > self.max:
            G = self.max
        if B > self.max:
            B = self.max
        total = np.max([R,G,B])
        R = self.div(R, total)
        G = self.div(G, total)
        B = self.div(B, total)
        A = total / self.max
        glColor4f(abs(B), abs(R), abs(G),A) #check here    
        glVertex3fv(pos) #center circle        
        for i in range(40):
            px = radius * math.cos(i) + x
            py = radius * math.sin(i) + y
            v = [px,py, pos[2]]
            glColor4f(abs(B), abs(R), abs(G),0)
            glVertex3fv(v)
        glEnd()
        glPointSize(8)
        glBegin(GL_POINTS)    #GL_POINTS
        glColor3f(0, 0, 0)   
        glVertex3fv(pos)
        glEnd()


    # def circle(self, radius, posB, norB,  force):
    #     x = posB[0]
    #     y = posB[1]
    #     glBegin(GL_TRIANGLE_FAN)
    #     glColor4f(abs(norB[2]), abs(norB[0]), abs(norB[1]), force / self.max)     
    #     glVertex3fv(posB) #center circle        
    #     for i in range(20):
    #         px = radius * math.cos(i) + x
    #         py = radius * math.sin(i) + y
    #         v = [px,py, posB[2]]
    #         glColor4f(abs(norB[2]), abs(norB[0]), abs(norB[1]), 0)
    #         glVertex3fv(v)
    #     glEnd()
    #     glPointSize(1)
    #     glBegin(GL_POINTS)    #GL_POINTS
    #     glColor3f(0, 0, 0)   
    #     glVertex3fv(posB)
    #     glEnd()

    def draw(self):
        glClearColor(1,1,1,1)
        glClear(GL_COLOR_BUFFER_BIT)
        glLoadIdentity()
        gluPerspective(self.fov, self.aspect, self.near, self.far)
        gluLookAt(self.cameraPos[0], self.cameraPos[1], self.cameraPos[2], self.target[0], self.target[1], self.target[2], self.up[0], self.up[1], self.up[2])
        for i in range(len(self.forces)):
            self.circle(0.03, self.forces[i], self.pos[i], self.weight)
        #for i in range(len(self.norB)):
        #    self.circle(0.06,self.posB[i], self.norB[i], self.force[i])
        glutSwapBuffers()
        glFlush()
        glutPostRedisplay()        