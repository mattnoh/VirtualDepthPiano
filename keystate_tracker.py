import cv2
import numpy as np

MIN_PRESS_TIME = 0.1


class KeyStateTracker:
    def __init__(self, num_keys):
        self.num_keys = num_keys
        self.finger_states = {}
        self.active_presses = set()
        
    def update_finger_state(self, finger_id, key_id, is_touching_desk, current_time):
        """Update the state of a specific finger on a specific key"""
        state_key = (finger_id, key_id)
        
        # Clean up states for fingers that are no longer on any key
        keys_to_remove = []
        for (fid, kid), state in self.finger_states.items():
            if fid == finger_id and kid != key_id and state['pressed']:
                # This finger moved away from a previously pressed key
                state['pressed'] = False
                # Remove from active presses if no other finger is pressing this key
                other_fingers_pressing = any(
                    self.finger_states.get((other_fid, kid), {}).get('pressed', False)
                    for other_fid in range(10) if other_fid != fid
                )
                if not other_fingers_pressing:
                    self.active_presses.discard(kid)
                keys_to_remove.append((fid, kid))
        
        # Remove old states
        for key_to_remove in keys_to_remove:
            if key_to_remove in self.finger_states:
                del self.finger_states[key_to_remove]
        
        if key_id is None:  # Finger is outside keyboard bounds
            return False
            
        if state_key not in self.finger_states:
            self.finger_states[state_key] = {'pressed': False, 'last_press_time': 0}
        
        state = self.finger_states[state_key]
        
        # Check for new press
        if not state['pressed'] and is_touching_desk:
            if current_time - state['last_press_time'] > MIN_PRESS_TIME:
                state['pressed'] = True
                state['last_press_time'] = current_time
                self.active_presses.add(key_id)
                return True
        
        # Check for release
        elif state['pressed'] and not is_touching_desk:
            state['pressed'] = False
            if key_id in self.active_presses:
                other_fingers_pressing = any(
                    self.finger_states.get((fid, key_id), {}).get('pressed', False)
                    for fid in range(10) if fid != finger_id
                )
                if not other_fingers_pressing:
                    self.active_presses.discard(key_id)
        
        return False
    
    def get_active_keys(self):
        return self.active_presses.copy()