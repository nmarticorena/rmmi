import spatialmath as sm
import spatialgeometry as sg
import json
import numpy as np
class Follower():
    def __init__(self, json_path, n_interp = 10) -> None:
        with open(json_path) as json_file:
            self.data = json.load(json_file)
    
        poses = self.get_cameras()
        self.n_views = len(poses)
        self.poses = list(self.smoothed_poses(poses, n_interp))
        self.current_index = 0
        self.last_pose = self.poses[0]
    
    def step(self):
        self.current_index += 1
        if self.current_index >= len(self.poses):
            self.current_index = 0
        # Check if it jumped by accident:
        distance = np.linalg.norm(self.poses[self.current_index].A[:3, -1] - self.last_pose.A[:3, -1])
        if distance > 0.5:
            self.current_index +=1
            self.step()
        # while self.poses[self.current_index].A[2,-1] < 0.1:
        #     self.current_index +=1
        self.last_pose = self.poses[self.current_index] 
        return self.poses[self.current_index]
    
    
    def smoothed_poses(self,poses, n_interp):
        for i in range(len(poses)-1):
            interp_poses = poses[i].interp(poses[i+1], n_interp)
            for pose in interp_poses:
                yield pose 
        
                    
    def get_cameras(self):
        cameras = []
        for camera in self.data["frames"]:
            pose = sm.SE3(np.array(camera["transform_matrix"]), check = False)
            # pose.norm()
            cameras.append(pose.norm())
        return cameras
    
class CreatePath(Follower):
    def __init__(self, json_path, n_interp = 10) -> None:
        super().__init__(json_path, n_interp)

    def get_cameras(self):
        cameras = []
        for camera in self.data["poses"]:
            pose = sm.SE3(camera)
            pose = pose.norm()
            cameras.append(pose)
        return cameras



    
if __name__ == "__main__":
    import swift
    from mm_neo.utils.swift_utils import load_mesh
    import numpy as np
   
    env = swift.Swift()
    env.launch(realtime=True)
    # env.add(gt_mesh)
    
    follow = CreatePath("configs/kitchen_1.json")
    cameras = follow.poses
    
    # for camera in cameras:
    #     ax = sg.Axes(0.1)
    #     ax.T = camera
    #     env.add(ax)
    #     env.step()
    
    ax = sg.Axes(0.1)
    env.add(ax)
    for i in range(len(follow.poses)):
        camera = follow.step()
        ax.T = camera
        print(ax.T)
        env.step()
        # time.sleep(0.001)
