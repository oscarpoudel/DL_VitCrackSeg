import cv2
import torch
import torchvision.transforms as transforms
from PIL import Image
import os
import numpy as np
import time

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the TorchScript model
model = torch.jit.load("model_traced.imp_pth", map_location=device)
model.eval()

# Define the transformation
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Resize to the input size that the model expects
    transforms.ToTensor(),  # Convert the PIL Image to a tensor
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize the image
])

# Start the video stream
cap = cv2.VideoCapture('http://10.0.0.94:8080/video')

# Directory to save the images
save_dir = 'segmented_images'
os.makedirs(save_dir, exist_ok=True)  

frame_rate = 1  # Desired frame rate to process the segmentation
prev = 0  # Variable to keep track of time
frame_count = 0  # Frame counter to save images with different names

while cap.isOpened():
    time_elapsed = time.time() - prev
    ret, frame = cap.read()
    if not ret:
        break

    if time_elapsed > 1./frame_rate:
        prev = time.time()
        
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(frame_rgb)
        transformed_frame = transform(pil_image).unsqueeze(0).to(device)

        with torch.no_grad():
            pred_mask = model(transformed_frame)
            pred_mask = torch.sigmoid(pred_mask)
            pred_mask = (pred_mask > 0.2).float()  # Binarize the output

        pred_mask_np = pred_mask.squeeze().cpu().numpy()
        pred_mask_np = cv2.resize(pred_mask_np, (frame.shape[1], frame.shape[0]))

        # Applying the mask where prediction > 0.5
        mask_overlay = (pred_mask_np > 0.2).astype(np.uint8)
        
        # Creating an overlay image with green where the mask is positive
        green_overlay = np.zeros_like(frame)
        green_overlay[mask_overlay == 1] = [0, 255, 0]
        
        cv2.addWeighted(frame, 0.5, green_overlay, 0.5, 0, frame)

        filename = f'{save_dir}/frame_{frame_count:04d}.jpg'
        cv2.imwrite(filename, frame)
        frame_count += 1

        cv2.imshow('Segmentation Overlay', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
