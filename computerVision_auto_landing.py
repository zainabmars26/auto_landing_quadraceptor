import cosysairsim as airsim
import cv2
import numpy as np
import torch
import matplotlib.pyplot as plt
from transformers import SegformerForSemanticSegmentation, SegformerImageProcessor
import time 
from professional_HUD import draw_professional_hud

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Loading SegFormer model on device: {device}")

import time
prev_time = time.time()


UAVID_PALETTE = np.array([
    [0, 0, 0],         # 0: Clutter
    [128, 0, 0],       # 1: Building
    [128, 64, 128],    # 2: Road
    [0, 128, 0],       # 3: Tree
    [128, 128, 0],     # 4: Vegetation
    [64, 0, 128],      # 5: Moving Car
    [192, 0, 192],     # 6: Static Car
    [64, 64, 0]        # 7: Human
], dtype=np.uint8)

processor = SegformerImageProcessor.from_pretrained("nvidia/mit-b0")
model = SegformerForSemanticSegmentation.from_pretrained("nvidia/mit-b0", num_labels=8)
model.load_state_dict(torch.load(r"assets\segformer_uavid_epoch_106_13-7.pt", map_location=device))
model.to(device).eval()

client = airsim.MultirotorClient()
client.confirmConnection()
client.enableApiControl(True)
client.armDisarm(True)

print("Taking off and ascending to 10 meters...")
client.takeoffAsync().join()
client.moveToZAsync(-40, 4).join() 

print("Drone is hovering. Starting live camera feed and prediction loop...")
print("Press 'q' in the OpenCV window to land the drone and exit.")

is_landing = False
text = "drone auto-landing test"
cluster_center_x = None
cluster_center_y = None 
centered = False 
image_width = 0
image_height = 0
stop_searching = False

target_founded = False
target_x = 0
target_y = 0

target_vx = 0.0
target_vy = 0.0
target_vz = 0.0

target_land_x = None
target_land_y = None

last_target_x = None
last_target_y = None

safe_spot_frozen = False
chosen_candidate = None


frame_counter = 0
cand1_safe = False
cand2_safe = False


def check_zone_safety(global_center, radius, road_mask):
    h_img, w_img = road_mask.shape[:2]
    gx, gy = global_center
    
    if not (0 <= gx < w_img and 0 <= gy < h_img):
        return False
        
    ymin, ymax = max(0, gy - radius), min(h_img, gy + radius)
    xmin, xmax = max(0, gx - radius), min(w_img, gx + radius)
    road_crop = road_mask[ymin:ymax, xmin:xmax]
    
    if road_crop.size == 0:
        return False
    road_pixel_ratio = np.mean(road_crop == 255) 
    return road_pixel_ratio > 0.90  

try:
    while True:
        state = client.getMultirotorState()
        current_z = state.kinematics_estimated.position.z_val

        responses = client.simGetImages([
            airsim.ImageRequest("downward_cam", airsim.ImageType.Scene, False, False)
        ])
        response = responses[0]

        

        img1d = np.frombuffer(response.image_data_uint8, dtype=np.uint8)
        img_rgb = img1d.reshape(response.height, response.width, 3)
        img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)

        image_width = img_bgr.shape[1]
        image_height = img_bgr.shape[0]
        
        center_x = image_width / 2
        center_y = image_height / 2
        cam_center_x = center_x
        cam_center_y = center_y

        inputs = processor(images=img_rgb, return_tensors="pt")
        pixel_values = inputs["pixel_values"].to(device)
        
        with torch.no_grad():
            outputs = model(pixel_values=pixel_values)
            logits = outputs.logits
            
        upsampled_logits = torch.nn.functional.interpolate(
            logits, size=(response.height, response.width), mode="bilinear", align_corners=False
        )
        probabilities = torch.nn.functional.softmax(upsampled_logits, dim=1).squeeze(0).cpu().numpy()
        
        prediction_indices = probabilities.argmax(axis=0)
        confidence_map = np.max(probabilities, axis=0)

        confidence_threshold = 0.95
        car_classes = (prediction_indices == 6)
        low_confidence_cars = car_classes & (confidence_map < confidence_threshold)

        if np.any(low_confidence_cars):
            alt_probabilities = probabilities.copy()
            alt_probabilities[6, low_confidence_cars] = -1
            next_best_classes = alt_probabilities.argmax(axis=0)
            prediction_indices[low_confidence_cars] = next_best_classes[low_confidence_cars]

        prediction_mask_rgb = UAVID_PALETTE[prediction_indices]
        prediction_mask_bgr = cv2.cvtColor(prediction_mask_rgb, cv2.COLOR_RGB2BGR)

        cv2.namedWindow("Drone Stream (Left) vs SegFormer Prediction Map (Right)", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Drone Stream (Left) vs SegFormer Prediction Map (Right)", 1280*2, 720)

        color_visualization = np.zeros_like(img_bgr)
        color_visualization[prediction_indices == 2] = [255, 255, 255] # Road
        color_visualization[prediction_indices == 6] = [192, 0, 192]   # Static Cars
        
        if not safe_spot_frozen:
            car_mask = (prediction_indices == 6).astype(np.uint8) * 255
            road_mask = (prediction_indices == 2).astype(np.uint8) * 255    

            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
            car_mask_cleaned = cv2.morphologyEx(car_mask, cv2.MORPH_CLOSE, kernel)

            contours, _ = cv2.findContours(car_mask_cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            min_dist = float('inf')
            max_confidence_for_tie = -1.0
            closest_cx, closest_cy = None, None
            closest_rect = None  
            closest_contour = None
            threshold_of_neglecting_area = (200*40)/abs(current_z) #pixel per meter
            print(f"threshold_of_neglecting_area = {threshold_of_neglecting_area}")

            for con in contours:
                if cv2.contourArea(con) < int(threshold_of_neglecting_area):
                    continue
                
                rect = cv2.minAreaRect(con)
                (cx_temp, cy_temp), (w_temp, h_temp), angle_temp = rect
                
                box = cv2.boxPoints(rect)
                box = np.int32(box)
                cv2.drawContours(img_bgr, [box], 0, (255, 255, 0), 1)
                
                if is_landing and last_target_x is not None and last_target_y is not None:
                    dist = np.hypot(cx_temp - last_target_x, cy_temp - last_target_y)
                else:
                    dist = np.hypot(cx_temp - cam_center_x, cy_temp - cam_center_y)
                
                mask_single_car = np.zeros(car_mask_cleaned.shape, dtype=np.uint8)
                cv2.drawContours(mask_single_car, [con], -1, 255, -1)
                car_pixels_confidence = confidence_map[mask_single_car == 255]
                current_car_conf = np.mean(car_pixels_confidence) if len(car_pixels_confidence) > 0 else 0.0

                distance_tolerance = 15.0
                is_significantly_closer = dist < (min_dist - distance_tolerance)
                #is_almost_same_dist_but_more_confident = (abs(dist - min_dist) <= distance_tolerance) and (current_car_conf > max_confidence_for_tie)

                if is_significantly_closer: #or is_almost_same_dist_but_more_confident:
                    min_dist = dist
                    max_confidence_for_tie = current_car_conf
                    closest_rect = rect
                    closest_contour = con
                    print(f"car pixel per meter {cv2.contourArea(closest_contour)}")
                    
                    M = cv2.moments(con)
                    if M["m00"] > 0:
                        closest_cx = M["m10"] / M["m00"]
                        closest_cy = M["m01"] / M["m00"]
                        
                    else:
                        closest_cx = cx_temp
                        closest_cy = cy_temp
                    
                
            box = cv2.boxPoints(closest_rect)
            box = np.int32(box)
            cv2.drawContours(img_bgr, [box], 0, (255, 0, 0), 1)
            car_found = False
            if closest_rect is not None:
                car_found = True
                (cx, cy), (w, h), angle = closest_rect
                cx, cy = closest_cx, closest_cy
                color_ = (255, 0, 0)
                cv2.circle(img_bgr, (int(cx), int(cy)), 5, color_, 2)

                last_target_x, last_target_y = int(cx), int(cy)

                if w < h:
                    car_angle = angle + 90
                    length, width = h, w
                else:
                    car_angle = angle
                    length, width = w, h
                    
                car_angle_rad = np.deg2rad(car_angle)
                margin = max(40, 0.20 * length)
                print(f"length = {length}")
                project_dist = (length / 2) + margin
                
                offset_x = project_dist * np.cos(car_angle_rad)
                offset_y = project_dist * np.sin(car_angle_rad)
                
                cand1_local = (int(cx + offset_x), int(cy + offset_y))
                cand2_local = (int(cx - offset_x), int(cy - offset_y))

                landing_radius = 10 
               
                cand1_safe = check_zone_safety(cand1_local, landing_radius, road_mask)
                cand2_safe = check_zone_safety(cand2_local, landing_radius, road_mask)

                color1 = (0, 255, 0) if cand1_safe else (0, 0, 255)
                cv2.circle(img_bgr, cand1_local, landing_radius, color1, 2)
                #cv2.putText(img_bgr, "C1", (cand1_local[0]-8, cand1_local[1]+4), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color1, 1)

                color2 = (0, 255, 0) if cand2_safe else (0, 0, 255)
                cv2.circle(img_bgr, cand2_local, landing_radius, color2, 2)

                if cand1_safe:
                    target_x, target_y = cand1_local
                    cv2.putText(img_bgr, "C1", (cand1_local[0]-8, cand1_local[1]+4), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color1, 1)

                elif cand2_safe:
                    target_x, target_y = cand2_local
                    cv2.putText(img_bgr, "C2", (cand2_local[0]-8, cand2_local[1]+4), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color2, 1)

                
                if is_landing and chosen_candidate == 'C1' and cand1_safe:
                    # If we already chose C1 and it's still safe, stick with it!
                    target_x, target_y = cand1_local
                    target_land_x, target_land_y = cand1_local
                elif is_landing and chosen_candidate == 'C2' and cand2_safe:
                    # If we already chose C2 and it's still safe, stick with it!
                    target_x, target_y = cand2_local
                    target_land_x, target_land_y = cand2_local
                
                # If no previous choice was made, or the previous choice became unsafe:
                elif cand1_safe and cand2_safe:
                    dist_c1 = np.hypot(cand1_local[0] - center_x, cand1_local[1] - center_y)
                    dist_c2 = np.hypot(cand2_local[0] - center_x, cand2_local[1] - center_y)
                    
                    if dist_c1 <= dist_c2:
                        target_x, target_y = cand1_local
                        target_land_x, target_land_y = cand1_local
                        if is_landing: chosen_candidate = 'C1'  # Lock choice
                    else:
                        target_x, target_y = cand2_local
                        target_land_x, target_land_y = cand2_local
                        if is_landing: chosen_candidate = 'C2'  # Lock choice
                        
                elif cand1_safe:
                    target_x, target_y = cand1_local
                    target_land_x, target_land_y = cand1_local
                    if is_landing: chosen_candidate = 'C1'
                elif cand2_safe:
                    target_x, target_y = cand2_local
                    target_land_x, target_land_y = cand2_local
                    if is_landing: chosen_candidate = 'C2'
                else:
                    target_x, target_y = int(cx), int(cy)
                    chosen_candidate = None  # Reset if everything is lost
            else:
                if not is_landing:
                    last_target_x, last_target_y = None, None
            
            img_center = (int(image_width / 2), int(image_height / 2))
            cv2.circle(img_bgr, img_center, 5, (0, 255, 0), -1)

            if not is_landing:
                text = "Flying..."
                client.moveByVelocityAsync(vx=6.0, vy=0, vz=0, duration=0.1)
            else:
                if car_found:
                    text = "TARGET ACQUIRED - INITIATING APPROACH"
                    cv2.circle(img_bgr, (target_x, target_y), 5, (255, 255, 0), -1)
                    
                    error_x = (target_x - center_x) / center_x
                    error_y = (center_y - target_y) / center_y

                    K = 2.5
                    vx = np.clip(K * error_y, -2.0, 2.0)
                    vy = np.clip(K * error_x, -2.0, 2.0)

                    centered = abs(target_x - center_x) < 30 and abs(target_y - center_y) < 30


                    if centered:
                        vz = 5.5  
                        vx = 0.0
                        vy = 0.0
                        if abs(current_z) < 12:
                            safe_spot_frozen=True
                    else:
                        vz = 0.0 

                    target_vx = vx
                    target_vy = vy
                    target_vz = vz

                    if abs(current_z) >30 :
                        vz=5
                        target_vz = vz
                        client.moveByVelocityBodyFrameAsync(
                        vx=vx, vy=vy, vz=3, duration=0.1
                    )
                    else:
                        client.moveByVelocityBodyFrameAsync(
                            vx=vx, vy=vy, vz=vz, duration=0.1
                        )

                    if current_z > -1.0:
                        print("Safe touchdown achieved.")
                        client.landAsync().join()
                        #break
                else:
                
                    text = "LANDING ENABLED: SEARCHING FOR STATIC CARS..."
                    target_vx, target_vy, target_vz = 6.0, 0.0, 0.0
                    # Move forward in the global/body frame to continue hunting for targets
                    client.moveByVelocityAsync(vx=6.0, vy=0, vz=0, duration=0.1)
        else:
            # Land
            
            print("Landing...")
            vz=5
            target_vz = vz
            client.moveByVelocityBodyFrameAsync(
                        vx=0, vy=0, vz=5, duration=0.1
                    )
                    
            if current_z > -0.1: 
                client.armDisarm(False)
                client.enableApiControl(False)
                break
            

     
        current_error_x = target_x - center_x
        current_error_y = center_y - target_y
        img_bgr_hud = img_bgr.copy() 
        img_bgr_hud = draw_professional_hud(
            frame=img_bgr_hud,
            target_x=target_x,
            target_y=target_y,
            error_x=current_error_x,
            error_y=current_error_y,
            current_z=current_z,
            vx=target_vx,
            vy=target_vy,
            vz=target_vz,
            is_landing=is_landing,
            centered=centered
        )
        current_time = time.time()
        time_diff = current_time - prev_time
        
        # Avoid division by zero if a frame processes instantly
        fps = 1.0 / time_diff if time_diff > 0 else 0.0
        
        # Save the current time for the next frame's calculation
        prev_time = current_time

        # --- DRAW FPS ON THE SCREEN ---
        print( f"FPS: {int(fps)}")


        display_window = np.hstack((img_bgr_hud, prediction_mask_bgr))
        cv2.putText(display_window, text, (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2, cv2.LINE_AA)
        
        
        cv2.imshow("Drone Stream (Left) vs SegFormer Prediction Map (Right)", display_window)

      


        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            is_landing = True
            text = "Landing..."
            client.moveByVelocityAsync(vx=0, vy=0, vz=0, duration=1.0).join()

except Exception as e:
    print(f"An error occurred: {e}")
finally:
    client.enableApiControl(False)
    cv2.destroyAllWindows()