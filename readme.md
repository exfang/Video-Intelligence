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












