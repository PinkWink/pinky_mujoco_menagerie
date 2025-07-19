import mujoco as mj
from mujoco.glfw import glfw

import sys, os
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), "..")))

from utils.scene_creator import SceneCreator
from utils.mujoco_renderer import MuJoCoViewer
from utils.camera_recorder import CameraRecorder

class PinkyMuJoCo:
    def __init__(self):
        background, robot, objects = "floor_sky", "pinky", ["can", "milk", "lemon"]
        base_env_path, robot_path, object_paths = self.get_xml_paths(background, robot, objects)
        
        objects_to_spawn = [
            {"path": object_paths[0], "name": objects[0], "pos": "0.5 0.3 0.05"},
            {"path": object_paths[1], "name": objects[1], "pos": "0.5 0.0 0.05"},
            {"path": object_paths[2], "name": objects[2], "pos": "0.5 -0.3 0.05"},
        ]
        
        model = self.build_model_from_xmls(base_env_path, robot_path, objects_to_spawn)
        if model is None:
            exit()
        self.data = mj.MjData(model)
        
        self.simulator = MuJoCoViewer(model, self.data)
        window_robot = self.simulator.get_window_robot()

        robot_view_width, robot_view_height = glfw.get_framebuffer_size(window_robot)
        output_path = "record.mp4"

        self.recorder = CameraRecorder(
            window=window_robot,
            width=robot_view_width, 
            height=robot_view_height, 
            filename=output_path,
        )
        
    def get_xml_paths(self, background, robot, objects):
        script_path = os.path.abspath(__file__)
        scripts_dir = os.path.dirname(script_path)
        PROJECT_ROOT = os.path.dirname(scripts_dir)
        assets_dir = os.path.join(PROJECT_ROOT, "assets")

        base_env_path = os.path.join(assets_dir, "scenes", background + ".xml")
        robot_path = os.path.join(assets_dir, "robots", robot, robot + ".xml")
        object_paths = [
            os.path.join(assets_dir, "objects", objects[0] + ".xml"),
            os.path.join(assets_dir, "objects", objects[1] + ".xml"),
            os.path.join(assets_dir, "objects", objects[2] + ".xml"),
        ]
        return base_env_path, robot_path, object_paths
    
    def build_model_from_xmls(self, base_env_path, robot_path, objects_to_spawn):
        scene_xml_string = SceneCreator.build_mjcf_scene(
            base_env_path=base_env_path,
            robot_path=robot_path,
            objects_to_spawn=objects_to_spawn,
            save_xml=False
        )
        try:
            model = mj.MjModel.from_xml_string(scene_xml_string)
            return model
        except Exception as e:
            print(f"XML load failed: {e}")
            with open("debug_scene.xml", "w", encoding='utf-8') as f:
                f.write(scene_xml_string)
            print("For debug MJCF created 'debug_scene.xml'")
            return None
        
    def start_rendering(self):
        try:
            while not self.simulator.should_close():
                time_prev = self.data.time
                while (self.data.time - time_prev < 1.0 / 60.0):
                    self.simulator.step_simulation()
                    
                self.simulator.render_main(overlay_type="imu") # or robot_pose
                self.simulator.render_robot()
                self.simulator.poll_events()
                self.recorder.capture_and_write()
        except Exception as e:
            print(f"\n시뮬레이션을 종료합니다. {e}")
        finally:
            self.simulator.terminate()
            self.recorder.release()

if __name__ == "__main__":
    pinky_mj = PinkyMuJoCo() 
    pinky_mj.start_rendering()