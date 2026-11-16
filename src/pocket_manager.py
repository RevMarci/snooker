import cv2

class PocketManager:
    def __init__(self, frame_width, frame_height):
        
        self.pockets = {
            'top_left': (60, 60),
            'top_middle': (frame_width // 2, 45),
            'top_right': (frame_width - 60, 60),
            'bottom_left': (60, frame_height - 60),
            'bottom_middle': (frame_width // 2, frame_height - 45),
            'bottom_right': (frame_width - 60, frame_height - 60)
        }
        
        
        self.pocket_radius = 25

    def draw_pockets(self, overlay):
        
        for name, pos in self.pockets.items():
            # Magenta színű körök jelzik a lyukakat
            cv2.circle(overlay, pos, self.pocket_radius, (255, 0, 255), 2)
            cv2.circle(overlay, pos, 3, (255, 0, 255), -1) # Középpont
            cv2.putText(overlay, name, (pos[0] - 30, pos[1] - 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 255), 1)

    def check_ball_in_pocket(self, ball_center):
        """
        Ellenőrzi, hogy egy golyó koordinátája (ball_center: (x, y)) 
        belelóg-e valamelyik lyuk zónájába.
        """
        bx, by = ball_center
        for name, (px, py) in self.pockets.items():
            
            distance = ((bx - px) ** 2 + (by - py) ** 2) ** 0.5
            if distance <= self.pocket_radius:
                return True, name 
        
        return False, None