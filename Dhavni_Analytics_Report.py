#!/usr/bin/env python
# coding: utf-8

# # Flowchart

# https://lucid.app/lucidchart/90a351a5-823c-447a-97b3-85059c5c2968/edit?viewport_loc=-952%2C-181%2C3971%2C1865%2C0_0&invitationId=inv_82a4ea61-31c4-433f-9b38-c9e1104a96c0

# # Algorithm

# In[2]:


import cv2
import numpy as np

# Load the reference (good) donut circle image
good_donut = cv2.imread('good.png', cv2.IMREAD_COLOR)

# Function to detect and classify defects in a given image
def detect_defects(current_image):
    # Ensure the current image has the same dimensions as the reference image
    current_image = cv2.resize(current_image, (good_donut.shape[1], good_donut.shape[0]))

    # Calculate the absolute difference between the current and reference images
    diff_image = cv2.absdiff(good_donut, current_image)
    
    # Convert the difference image to grayscale (8-bit single-channel)
    gray_diff_image = cv2.cvtColor(diff_image, cv2.COLOR_BGR2GRAY)
    
    # Apply a threshold to highlight differences (you may need to adjust this threshold)
    threshold = 30
    _, thresholded_image = cv2.threshold(gray_diff_image, threshold, 255, cv2.THRESH_BINARY)
    
    # Find contours in the thresholded image
    contours, _ = cv2.findContours(thresholded_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    defects = {
        'flashes': [],
        'cuts': []
    }
    
    # Create a copy of the current image for visualization
    result_image = current_image.copy()
    
    for contour in contours:
        # Filter out small noise regions
        if cv2.contourArea(contour) < 100:
            continue

        # Calculate the bounding box of the contour
        x, y, w, h = cv2.boundingRect(contour)

        # Check if the defect is inside the donut circle or on the outer edge
        if (x + w // 2 < current_image.shape[1] // 2):
            defects['cuts'].append({
                'x': x,
                'y': y,
                'width': w,
                'height': h
            })
            cv2.rectangle(result_image, (x, y), (x + w, y + h), (0, 0, 255), 2)  # Draw red rectangle for cuts
        else:
            defects['flashes'].append({
                'x': x,
                'y': y,
                'width': w,
                'height': h
            })
            cv2.rectangle(result_image, (x, y), (x + w, y + h), (0, 255, 0), 2)  # Draw green rectangle for flashes
    
    return defects, result_image

# Load and process four defective images using a loop
defective_images = ['defect1.png', 'defect2.png', 'defect3.png', 'defect4.png']

for image_file in defective_images:
    # Load the current defective image
    current_image = cv2.imread(image_file, cv2.IMREAD_COLOR)
    
    # Call the detect_defects function to detect defects and get result_image
    defects, result_image = detect_defects(current_image)
    
    # Display the result_image with detected defects
    cv2.imshow(f"Defects in {image_file}", result_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    # Print the detected defects for each image
    print(f"Defects in {image_file}:")
    print("Flashes:")
    for flash in defects['flashes']:
        print(f"Location: ({flash['x']}, {flash['y']}), Size: ({flash['width']}, {flash['height']})")
    print("Cuts:")
    for cut in defects['cuts']:
        print(f"Location: ({cut['x']}, {cut['y']}), Size: ({cut['width']}, {cut['height']})")
    print("\n")


# # Dynamic_System

# In[3]:


import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt


# Define the system of differential equations
def system(t, variables):
    x, y, z = variables
    a = 1.0  # Replace with your desired values for a, b, and c
    b = 2.0
    c = 3.0
    dxdt = a * (y - b)
    dydt = b * x - y - x * z
    dzdt = x * y - c * z
    return [dxdt, dydt, dzdt]


# Define the time span for the integration
t_span = (0, 10)  # Adjust the time range as needed


# Define initial conditions
initial_conditions = [1.0, 2.0, 3.0]  # Replace with your desired initial values for x, y, and z


# Solve the system of differential equations
sol = solve_ivp(system, t_span, initial_conditions, t_eval=np.linspace(t_span[0], t_span[1], 1000))


# Extract the solutions
t = sol.t
x = sol.y[0]
y = sol.y[1]
z = sol.y[2]


# Plot the solutions
plt.figure(figsize=(10, 6))
plt.plot(t, x, label='x(t)')
plt.plot(t, y, label='y(t)')
plt.plot(t, z, label='z(t)')
plt.xlabel('Time (t)')
plt.ylabel('Values')
plt.title('Solutions to the System of Differential Equations')
plt.legend()
plt.grid(True)
plt.show()


# In[ ]:




