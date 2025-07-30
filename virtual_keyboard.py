import cv2
import numpy as np

class VirtualKeyboard:
    def __init__(self, center_x, center_y, width, height, num_keys, key_spacing=10):
        self.center_x = center_x
        self.center_y = center_y
        self.width = width
        self.height = height
        self.num_keys = num_keys
        self.key_spacing = key_spacing
        
        # Calculate keyboard bounds
        self.left = center_x - width // 2
        self.right = center_x + width // 2
        self.top = center_y - height // 2
        self.bottom = center_y + height // 2
        
        # Calculate key dimensions with spacing
        total_spacing = (num_keys - 1) * key_spacing
        available_width = width - total_spacing
        self.key_width = available_width // num_keys
        
    def get_key_bounds(self, key_idx):
        """Get the bounds of a specific key"""
        key_left = self.left + (key_idx * (self.key_width + self.key_spacing))
        key_right = key_left + self.key_width
        return key_left, self.top, key_right, self.bottom
    
    def is_point_in_keyboard(self, x, y):
        """Check if a point is within any key bounds (including spacing)"""
        if not (self.top <= y <= self.bottom):
            return False
        
        # Check if point is within any key
        for i in range(self.num_keys):
            key_left, _, key_right, _ = self.get_key_bounds(i)
            if key_left <= x <= key_right:
                return True
        return False
    
    def get_key_from_point(self, x, y):
        """Get the key index from a point, or None if outside keyboard"""
        if not (self.top <= y <= self.bottom):
            return None
        
        # Check each key individually
        for i in range(self.num_keys):
            key_left, _, key_right, _ = self.get_key_bounds(i)
            if key_left <= x <= key_right:
                return i
        
        return None  # Point is in spacing between keys
    
    def draw(self, frame, active_keys=None):
        """Draw the virtual keyboard on the frame"""
        if active_keys is None:
            active_keys = set()
        
        # Draw keyboard background (semi-transparent)
        overlay = frame.copy()
        cv2.rectangle(overlay, (self.left - 20, self.top - 20), (self.right + 20, self.bottom + 20), 
                    (30, 30, 30), -1)
        cv2.addWeighted(overlay, 0.3, frame, 0.7, 0, frame)
        
        # Draw individual keys with spacing (semi-transparent)
        for i in range(self.num_keys):
            key_left, key_top, key_right, key_bottom = self.get_key_bounds(i)

            if i in active_keys:
                key_color = (0, 255, 0)  # Bright green for active keys
                border_color = (0, 200, 0)
                alpha = 0.9
            else:
                key_color = (240, 240, 240)  # Light gray for inactive keys
                border_color = (150, 150, 150)
                alpha = 0.4  # Lower alpha for more transparency

            # Draw key rectangle on an overlay
            key_overlay = frame.copy()
            cv2.rectangle(key_overlay, (key_left, key_top), (key_right, key_bottom), key_color, -1)
            cv2.addWeighted(key_overlay, alpha, frame, 1 - alpha, 0, frame)

            # Draw key border
            cv2.rectangle(frame, (key_left, key_top), (key_right, key_bottom), border_color, 3)

            # Draw key number
            text = str(i + 1)
            text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 1.0, 2)[0]
            text_x = key_left + (self.key_width - text_size[0]) // 2
            text_y = key_top + (self.height + text_size[1]) // 2
            text_color = (0, 0, 0) if i not in active_keys else (255, 255, 255)
            cv2.putText(frame, text, (text_x, text_y),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, text_color, 2)