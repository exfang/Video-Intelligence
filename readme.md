# Project Background: Intelligent Video Analytics


In the industry of healthcare and caregiving, the need for advanced technologies that can enhance patient monitoring and support is ever-growing. The Intelligent Video Analytics project emerges as a solution that leverages the power of video capturing and state-of-the-art YOLOv8 pretrained models to create real-time detection models tailored for specific healthcare use cases.

# Technological Foundation: YOLOv8 Pretrained Models and Video Capturing


The foundation of our project lies in the integration of YOLOv8 pretrained models, renowned for their efficiency and accuracy in object detection tasks. These models are seamlessly integrated with video capturing systems, allowing for continuous monitoring of patients in real time.

# Use Cases: Fall Detection and Medication Consumption



The Intelligent Video Analytics project addresses two crucial use cases in healthcare:

1. **Fall Detection**:
One of the primary objectives is the real-time detection of patient falls. By employing advanced computer vision techniques, the system can swiftly identify instances of patients falling, enabling immediate and targeted support. This proactive approach aims to significantly reduce response times in critical situations.

2. **Medication Consumption Detection**:
Timely and proper medication consumption is vital for effective healthcare management. The project focuses on developing models that can detect whether patients are adhering to their medication schedules. This functionality not only ensures medication compliance but also provides a tool for caregivers to remotely supervise and intervene when necessary.

# Project Goal: Alleviating Caregiver Burden through Automation


The overarching goal of the Intelligent Video Analytics project is to reduce the burden on caregivers. By automating the supervision of medication consumption and fall detection, caregivers are empowered with real-time insights into the well-being of their patients. This automation not only enhances the efficiency of healthcare monitoring but also allows caregivers to focus on more personalized and critical aspects of patient care.

# Anticipated Benefits:



1. **Real-time Support**:
Swift detection of falls enables immediate support, potentially preventing further complications and reducing the severity of injuries.

2. **Medication Adherence**:
The system provides caregivers with tools to ensure patients are adhering to their medication schedules, promoting better health outcomes.

3. **Reduced Workload**:
Automation of supervision tasks significantly reduces the workload on caregivers, allowing them to allocate their time and attention more effectively.

# Features



## Main Webapp Features:
- Fall/Medication consumption detection
- Dashboard of historic fall/medication consumption

## Fall Detection Specific Features:
- Setting up of caregiver information
- Backlog of patient fall history
- Alert of real-time fall detected

## Medication Consumption Specific Features:
- Pill Upload and Configuration
- Button to retrain model based on uploaded pills
- Medication Routine Configuration
- Backlog of patient medication consumption

## Additional Features:
- Memory Game - drag and drop user uploaded images
- Login system


# Getting Started


How to run the webapp locally:
1. Clone repository into VS Code
2. Create a virtual environment in VS Code "cmd" terminal 
   - `python -m venv venv`
3. Change directory to the virtual environment in VS Code "cmd" terminal 
   - `cd venv\Scripts`
4. Activate the virtual environment when in the "Scripts" directory 
   - `activate`
5. Return to the 'FSP_IVideoAnalyics' directory
   - `cd ../..`
6. Within the 'FSP_IVideoAnalyics' directory, download the necessary libraries 
   - `pip install -r /path/to/requirements.txt`
7. After the libraries have been downloaded, change directory to the root folder of the webapp. 
   - `cd ./FSP_IVideoAnalytics/Webapp`
8. Run the webapplication
   - 'python main.py'



# User Guide:
## Fall Detection
1. Head over to the 'Monitor' Page to access the Live feed, where fall detection takes place as well.
   - Falls that are identified from the Deep Learning Model will have the table on the side updated as well as the one in 'Falls Backlog under 'Fall info'.
   - Analytics to visualise the dashboard can be found under 'Fall Analytics'.

2. In the 'Falls Backlog' page, a table which records information collected from the fall is documented. Data collected includes:
   - Fall time and date
   - Image of fall occurrence
   - Confidence score of the model
   - Delete button (Remove row)

3. In the 'Caregiver' page, users are able to add caregiver details to be notified. Information of caregivers include:
   - Name
   - Image of caregiver
   - Number
   - Email
   - Delete button (Remove Row)

4. In the 'Falls Analytics' page, a dashboard which visualises the information found in fall backlog is displayed. The dashboard includes:
   - Slider filter for date of fall and confidence
   - Line Chart Visual to display relationship between fall date and confidence to show the trend
   - Bar Chart to show the number of falls based on the filtered results for date and confidence range.


## Pill Detection

1. Setting up the pill detection model
   - Upload images of pills patients are required to consume. (Manage Pills > Upload Pills)
   - Configure and ensure the right pills are uploaded. (Manage Pills > Pill Configuration)
   - Once you verify that the right pills are uploaded, press 'Retrain Model'. (Manage Pills > Pill Configuration)
   - While the model is training, configure the patients' medication routine. (Manage Pills > Medication Routine Configuration)

2. Using the pill detection model
   - Access the pill detection page. (Pill Detection)
   - Following the routines configured and displayed in the table on the right, ensure the right pills are detected on the camera, left panel. (Pill Detection)
   - When all pills are detected, a button will pop up allowing the user to 'Record Video' of themselves eating the pill. (Pill Detection)
   - Users will be redirected to the 'Medication Recording' page, which can only be accessed via the Pill Detection page (Manage Pills > Medication Recording)
   - Once the users are done recording themselves eating the pills, they can check the recording. (Manage Pills > View Medication Recording)

3. Dashboard Management
   - The dashboard uses videos from the (Manage Pills > View Medication Recording) page and routines stored in (Manage Pills > Medication Routine History).
   - Users can tidy up the dashboard by deleting past routines/recordings if they deem them to be wrong/irrelevant.






# Use Cases 
Some Use Case examples of the Project:

## Use Case 1 : Elderly Care Facilities
> In an elderly care facility, the Intelligent Video Analytics system is deployed to monitor the well-being of residents.

### Fall Detection:
- A resident, Mrs. Lee, experiences a sudden fall in her room.
- The system detects the fall in real-time through video analysis.
- Caregivers receive an immediate alert on their devices, allowing them to respond promptly and provide assistance.

### Medication Consumption Detection:
- Mr. Tan has a daily medication routine.
- The system monitors his medication consumption patterns using video analysis.
- If there's a deviation from the prescribed schedule, caregivers are notified, enabling them to ensure Mr. Tan takes his medications on time.


## Use Case 2: Remote Patient Monitoring
> Patients recovering at home after surgery or medical procedures are monitored remotely using the Intelligent Video Analytics system.

### Fall Detection:
- John, a post-surgery patient, experiences dizziness and falls in his living room.
- The system detects the fall and alerts the healthcare provider.
- The healthcare provider contacts John to assess his condition and dispatches help if needed.

### Medication Consumption Detection:
- Sarah is prescribed a specific medication regimen for post-operative care.
- The system monitors her medication adherence through video analysis.
- If Sarah forgets to take her medication, the system sends a notification to her and her designated caregiver.


## Use Case 3: Home Healthcare
> Patients receiving home healthcare services have the Intelligent Video Analytics system installed for enhanced monitoring.

### Fall Detection:
- Mr. and Mrs. Chang, elderly patients receiving home healthcare, are living independently.
- The system detects a fall in their kitchen and immediately notifies the healthcare provider.
- A nurse is dispatched to assess the situation and provide necessary assistance.

### Medication Consumption Detection:
- Emma, a patient with a chronic condition, has a complex medication schedule.
- The system ensures that Emma follows her medication routine through video analysis.
- If any deviations are detected, the healthcare provider intervenes to address the issue promptly.


## Preview of Pill Detection Feature created by me.

Below is an image of the pill detection detecting the pills shown. The right contains the pills required to be consumed by the elderly. It is configured by the caregiver.

![image](https://github.com/exfang/Video-Intelligence/assets/98097965/492c9ae2-347e-4909-9498-195eaa7f1910)

Below is the pill upload page for caregivers to upload images of pills the elderly is required to consume.

![image](https://github.com/exfang/Video-Intelligence/assets/98097965/427a4d18-722b-4660-b3c6-7661205e0c43)

Caregivers can manage the pills on the pill configuration page.

![image](https://github.com/exfang/Video-Intelligence/assets/98097965/25fec43f-834d-4334-b821-ba07db8528d6)

When users are done uploading the pills, they MUST retrain the model to ensure it learns to detect the pills shown. I have done the backend for automating training with just the click of a button. Model training will run in the backend. 

![image](https://github.com/exfang/Video-Intelligence/assets/98097965/5256e4f7-b118-42dd-8d46-89ab4b79d9af)

Caregivers can proceed to configure medication routines

![image](https://github.com/exfang/Video-Intelligence/assets/98097965/85b9a46c-cf3b-449b-a011-b4a2fbb54aa5)

Each time a user shows the pills <I>they are required to consume</i> to the pill detection page, they are required to create a video log showing that they consumed the pills.

![image](https://github.com/exfang/Video-Intelligence/assets/98097965/27767186-6a64-41ff-ac42-d0b07eba7a01)

The video log gets stored along with an image to capture the shown pills.

![image](https://github.com/exfang/Video-Intelligence/assets/98097965/7c37e7db-bb1c-4bbc-8919-4c367cb13be3)

A dashboard is also created for caregivers to easily supervise whether their patients adhere to their medication routines.

![image](https://github.com/exfang/Video-Intelligence/assets/98097965/8ab0fb0f-f441-42ad-87c8-85ad6aab6978)

## Preview of Memory Drag-and-drop Game Feature created by me.

Similar to pill detection, caregivers can upload images of a person/object.

![image](https://github.com/exfang/Video-Intelligence/assets/98097965/758f2177-90ef-4d82-b5f5-fef6ba4c72e3)

Here are some existing configurations. Caregivers can manage the uploaded images.

![image](https://github.com/exfang/Video-Intelligence/assets/98097965/5dfe11ed-18ff-4296-a25f-2348845039b3)

The patient starts by pressing the start game and will be shown the images configured by the caregiver. 

![image](https://github.com/exfang/Video-Intelligence/assets/98097965/3e9b3f68-be18-4ca8-8976-c657a0c078b2)

When all images are placed in the correct labels, users can end the game and the results are logged.

![image](https://github.com/exfang/Video-Intelligence/assets/98097965/ab87535b-1450-4838-8721-a74b3a60de64)

A game history page is created for caregivers to evaluate the cognitive ability of a patient overtime. More tries with less photo = patient struggle to associate the photo with the correct labels, resulting in them having to take more tries than normal to correctly label all the images.

![image](https://github.com/exfang/Video-Intelligence/assets/98097965/6591d869-fa19-489c-abf8-948e9b94d18a)








