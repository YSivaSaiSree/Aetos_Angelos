#!/usr/bin/env python
# coding: utf-8

# In[1]:


#communication with location
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
from torchvision.transforms import ToTensor, Normalize, Resize, Compose
import cv2
import numpy as np
import requests
from geolite2 import geolite2

# Define data transforms
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# Load the trained model
model = models.resnet18(pretrained=False)
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, 4)  # Assuming 4 labels/classes: 0, 100, 108, 112
model.load_state_dict(torch.load('C:/Users/91630/Downloads/2resnet_model.pth', map_location=torch.device('cpu')))
model.eval()

cap = cv2.VideoCapture(0)

# Define the output text for each gesture label
output_texts = {
    0: 'No Gesture Detected',
    1: 'Gesture 100 Detected',
    2: 'Gesture 108 Detected',
    3: 'Gesture 112 Detected'
}

# Elastic Email API credentials
elastic_email_api_key = '41E26C631CB2D39A01355246745E0C2E033CBFCBD5D5A599644A36FEB523D56976DEB5CB816FB5C7A77B20A04A880862'
elastic_email_sender = 'sivasaisreeyakkala@gmail.com'
recipient_email = '21211a7263@bvrit.ac.in'

# Define the counter for captured gestures
capture_image = False
capture_counter = 0

# Initialize GeoIP reader
reader = geolite2.reader()

while True:
    ret, frame = cap.read()

    # Convert the frame to PIL Image format
    image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    # Apply transformations to the image
    preprocessed_image = transform(image)
    preprocessed_image = preprocessed_image.unsqueeze(0)

    with torch.no_grad():
        # Perform inference on the preprocessed image
        outputs = model(preprocessed_image)
        _, predicted = torch.max(outputs.data, 1)
        predicted_label = predicted.item()

    output_text = output_texts.get(predicted_label, 'Unknown Gesture')

    # Display the predicted label on the frame
    cv2.putText(frame, output_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.imshow('Camera', frame)
    if predicted_label in [100, 108, 112]:
        capture_image = True

    # Capture the image when capture_image is True or 'c' key is pressed
    if capture_image or (cv2.waitKey(1) & 0xFF == ord('c')):
        # Save the captured image with a unique filename
        capture_counter += 1
        filename = f"gesture_capture_{capture_counter}.jpg"
        cv2.imwrite(filename, frame)
        print(f"Gesture captured as {filename}")

        # Display the captured image
        captured_image = cv2.imread(filename)
        cv2.imshow('Captured Image', captured_image)
        
        # Get geolocation information based on IP address
        ip_address = requests.get('https://api.ipify.org').text
        location = reader.get(ip_address)

        # Send the alert email based on the predicted label
        if predicted_label == 2 and location and 'location' in location:
            subject = 'Gesture Detected: {}'.format(output_text)
            body = 'Gesture Detected: {}. Send ambulance assistance!'.format(output_text)
            location_info = location['location']
            latitude = location_info.get('latitude', 'Unknown')
            longitude = location_info.get('longitude', 'Unknown')

            body += f'\n\nLocation: Latitude: {latitude}, Longitude: {longitude}'
            
            # Send the alert email using Elastic Email API
            data = {
                'from': elastic_email_sender,
                'to': recipient_email,
                'subject': subject,
                'body': body,
                'isTransactional': True,
                'apikey': elastic_email_api_key
            }

            response = requests.post('https://api.elasticemail.com/v2/email/send', data=data)
            print(response.text)

        elif predicted_label == 1 and location and 'location' in location:
            subject = 'Gesture Detected: {}'.format(output_text)
            body = 'Gesture Detected: {}. Send police assistance!'.format(output_text)
            location_info = location['location']
            latitude = location_info.get('latitude', 'Unknown')
            longitude = location_info.get('longitude', 'Unknown')

            body += f'\n\nLocation: Latitude: {latitude}, Longitude: {longitude}'
            
            # Send the alert email using Elastic Email API
            data = {
                'from': elastic_email_sender,
                'to': recipient_email,
                'subject': subject,
                'body': body,
                'isTransactional': True,
                'apikey': elastic_email_api_key
            }

            response = requests.post('https://api.elasticemail.com/v2/email/send', data=data)
            print(response.text)

        elif predicted_label == 3 and location and 'location' in location:
            subject = 'Gesture Detected: {}'.format(output_text)
            body = 'Gesture Detected: {}. Send both police and ambulance assistance!'.format(output_text)
            location_info = location['location']
            latitude = location_info.get('latitude', 'Unknown')
            longitude = location_info.get('longitude', 'Unknown')

            body += f'\n\nLocation: Latitude: {latitude}, Longitude: {longitude}'
            
            # Send the alert email using Elastic Email API
            data = {
                'from': elastic_email_sender,
                'to': recipient_email,
                'subject': subject,
                'body': body,
                'isTransactional': True,
                'apikey': elastic_email_api_key
            }

            response = requests.post('https://api.elasticemail.com/v2/email/send', data=data)
            print(response.text)
            
        elif predicted_label == 0 and location and 'location' in location:
            subject = 'Gesture Detected: {}'.format(output_text)
            body = 'Gesture Detected: {}. No Gesture Identified!'.format(output_text)
            location_info = location['location']
            latitude = location_info.get('latitude', 'Unknown')
            longitude = location_info.get('longitude', 'Unknown')

            body += f'\n\nLocation: Latitude: {latitude}, Longitude: {longitude}'
            
            # Send the alert email using Elastic Email API
            data = {
                'from': elastic_email_sender,
                'to': recipient_email,
                'subject': subject,
                'body': body,
                'isTransactional': True,
                'apikey': elastic_email_api_key
            }

            response = requests.post('https://api.elasticemail.com/v2/email/send', data=data)
            print(response.text)


        # Reset capture_image back to False
        capture_image = False

    # Exit the loop when 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the camera and close all windows
cap.release()
cv2.destroyAllWindows()
#final

