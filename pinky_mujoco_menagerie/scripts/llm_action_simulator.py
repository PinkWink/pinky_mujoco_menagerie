import mujoco as mj

import queue
import re
import time

import sys, os
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), "..")))

from utils.scene_creator import SceneCreator
from utils.mujoco_renderer import MuJoCoViewer
from utils.object_detector import ObjectDetector

from llm.llm_runner import LLMRunner

ACTION_TABLE = {
    "멈춤": (0, 0),
    "직진": (10, 10),
    "후진": (-10, -10),
    "좌회전": (8, 10),
    "우회전": (10, 8),
    "제자리 회전": (10, -10),
}

class ModelBuilder:
    def __init__(self, background, robot, objects):
        base_env_path, robot_path, object_paths = self.get_xml_paths(background, robot, objects)
        
        objects_to_spawn = [
            {"path": object_paths[0], "name": objects[0], "pos": "0.5 0.3 0.05"},
            {"path": object_paths[1], "name": objects[1], "pos": "0.5 0.0 0.05"},
            {"path": object_paths[2], "name": objects[2], "pos": "0.5 -0.3 0.05"},
        ]
        
        self.model = self.build_model_from_xmls(base_env_path, robot_path, objects_to_spawn)
        
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
        
    def get_model(self):
        return self.model

class LLMActionSimulator:
    def __init__(self, command_queue=None):
        self.command_queue = command_queue
        self.action_end_sim_time = 0
        self.current_action = None
        
        background, robot, objects = "floor_sky", "pinky", ["can", "milk", "lemon"]
        model_builder = ModelBuilder(background, robot, objects)
        model = model_builder.get_model()
        if model is None:
            exit()
        self.data = mj.MjData(model)
        
        self.simulator = MuJoCoViewer(model, self.data)

    def move_robot(self, msg, duration=1):
        move_match = re.search(r"Action:\s*([^\n]+)", msg, re.IGNORECASE)
        robot_action = move_match.group(1).strip() if move_match else ""
        if robot_action not in ACTION_TABLE:
            return
        
        if robot_action in ["좌회전", "우회전"]:
            duration *= 1.6
        elif robot_action == "제자리 회전":
            duration *= 1.22
        else:
            duration = duration
            
        ctrl0, ctrl1 = ACTION_TABLE[robot_action]
        self.data.ctrl[0] = ctrl0
        self.data.ctrl[1] = ctrl1
        self.current_action = robot_action
        self.action_end_sim_time = self.data.time + duration

    def start_rendering(self):
        try:
            while not self.simulator.should_close():
                time_prev = self.data.time
                while (self.data.time - time_prev < 1.0 / 60.0):
                    self.simulator.step_simulation()
                
                if self.command_queue:
                    while not self.command_queue.empty():
                        msg = self.command_queue.get()
                        self.move_robot(msg)
                        
                # now = time.time()
                if self.current_action and self.data.time > self.action_end_sim_time:
                    self.data.ctrl[0] = 0
                    self.data.ctrl[1] = 0
                    self.current_action = None

                self.simulator.render_main(overlay_type="robot_pose")
                self.simulator.render_robot()
                self.simulator.poll_events()
                self.latest_frame = self.simulator.capture_img()

        except Exception as e:
            print(f"\n시뮬레이션을 종료합니다. {e}")
        finally:
            self.simulator.terminate()

if __name__ == "__main__":
    command_queue = queue.Queue()
    llm_action_sim = LLMActionSimulator(command_queue) 
    yolo = ObjectDetector(model_path="best.pt", conf=0.5)
    
    llm_runner = LLMRunner(prompt_path="../llm/prompts/prompt.yaml", model="gpt-4o-mini", command_queue=command_queue)
    llm_runner.start_thread(target=llm_runner.talk, args=(llm_action_sim, yolo))
    
    try:
        llm_action_sim.start_rendering()
    except KeyboardInterrupt:
        print("\nProgram interrupted by user.")
    finally:
        llm_runner.stop_event.set()
        if llm_runner.thread and llm_runner.thread.is_alive():
            llm_runner.thread.join(timeout=2.0)

        print("\nShutdown complete.")