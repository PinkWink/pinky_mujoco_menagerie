import mujoco as mj
from mujoco.glfw import glfw

class KeyboardCallbacks:
    def keyboardGLFW(self, window, key, scancode, act, mods, model, data):
        if act == glfw.PRESS or act == glfw.REPEAT:
            if key == glfw.KEY_W:  
                data.ctrl[0] = 3
                data.ctrl[1] = 3 
            elif key == glfw.KEY_S:
                data.ctrl[0] = 0  
                data.ctrl[1] = 0
            elif key == glfw.KEY_X: 
                data.ctrl[0] = -3
                data.ctrl[1] = -3
            elif key == glfw.KEY_D:
                data.ctrl[0] = 3
                data.ctrl[1] = -3
            elif key == glfw.KEY_A: 
                data.ctrl[0] = -3
                data.ctrl[1] = 3
            elif key == glfw.KEY_BACKSPACE: 
                mj.mj_resetData(model, data)
                mj.mj_forward(model, data)

        if act == glfw.RELEASE:
            if key in [glfw.KEY_W, glfw.KEY_S, glfw.KEY_A, glfw.KEY_D]:
                data.ctrl[0] = 0
                data.ctrl[1] = 0