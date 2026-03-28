import gymnasium as gym
import numpy as np
import mujoco
from gymnasium import spaces

class AICMuJoCoEnv(gym.Env):
    """Custom Environment that follows gym interface for AIC Cable Insertion"""
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30}

    def __init__(self, xml_path="../assets/scene.xml", render_mode=None):
        super().__init__()
        self.model = mujoco.MjModel.from_xml_path(xml_path)
        self.data = mujoco.MjData(self.model)
        self.render_mode = render_mode

        # Action Space: 6 Joint Velocities (scaled between -1 and 1)
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(6,), dtype=np.float32)

        # Observation Space: 6 Joints + 6 F/T + 7D Relative Pose Proxy = 19
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(19,), dtype=np.float32)

    def _get_obs(self):
        # 1. Proprioception: Joint Angles
        qpos = self.data.qpos[:6]
        
        # 2. Haptic Feedback: Axia80 Force/Torque
        force = self.data.sensor("wrist_force").data
        torque = self.data.sensor("wrist_torque").data
        
        # 3. Perception Proxy: Relative position of cable tip to socket
        tip_pos = self.data.site("cable_tip").xpos
        socket_pos = self.data.site("socket_center").xpos
        relative_pos = socket_pos - tip_pos
        
        # In a full setup, add the relative quaternion here to make it 7D
        dummy_quat = np.array([1.0, 0.0, 0.0, 0.0]) 

        return np.concatenate([qpos, force, torque, relative_pos, dummy_quat]).astype(np.float32)

    def step(self, action):
        # Apply action to motors
        self.data.ctrl[:6] = action * 50.0 # Scale action to actual motor commands
        
        # Step physics
        mujoco.mj_step(self.model, self.data)

        # Calculate Reward
        obs = self._get_obs()
        relative_pos = obs[12:15]
        distance = np.linalg.norm(relative_pos)
        
        # Dense reward for getting closer, penalty for excessive force
        force_magnitude = np.linalg.norm(obs[6:9])
        force_penalty = 0.1 * max(0, force_magnitude - 20.0) # Penalty if Force > 20N
        
        reward = -distance - force_penalty

        # Check Termination
        terminated = distance < 0.005 # Success condition
        if terminated:
            reward += 100.0 # Sparse success bonus
            
        truncated = False # Add time limit logic if needed

        return obs, reward, terminated, truncated, {}

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        mujoco.mj_resetData(self.model, self.data)
        
        # Optional: Add domain randomization to task board position here
        mujoco.mj_forward(self.model, self.data)
        
        return self._get_obs(), {}