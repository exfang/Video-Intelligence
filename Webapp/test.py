# @app.route('/medication_dashboard')
# def medication_dashboard():
#     # Fetch unique start/end date/time combinations from the database
#     unique_datetime_combinations = db.session.query(
#         Routine.start_date, Routine.end_date, Routine.start_time, Routine.end_time
#     ).group_by(
#         Routine.start_date, Routine.end_date, Routine.start_time, Routine.end_time
#     ).all()

#     # Convert the results to a list of dictionaries
#     unique_datetime_list = []

#     for start_date, end_date, start_time, end_time in unique_datetime_combinations:
#         current_date = datetime.now().date()
#         if end_date > current_date and current_date > start_date:
#             # If end date is in the future, calculate remaining medications
#             total_medications_required = (min(end_date, current_date) - max(start_date, current_date)).days + 1
#         else:
#             # If end date is in the past or today, calculate total medications required
#             total_medications_required = (end_date - start_date).days + 1 if start_date <= current_date else 0

#         unique_datetime_list.append({
#             'start_date': start_date,
#             'end_date': end_date,
#             'start_time': start_time,
#             'end_time': end_time,
#             'medication_count': 0,
#             'total_medications_required': total_medications_required
#         })

#     # Fetch image records from the recording_folder
#     image_records = []  # Replace this with your code to fetch image records

#     # Count the number of overlapping records for each routine
#     for datetime_combination in unique_datetime_list:
#         datetime_start = datetime.combine(datetime_combination['start_date'], datetime_combination['start_time'])
#         datetime_end = datetime.combine(datetime_combination['end_date'], datetime_combination['end_time'])

#         for image_record in image_records:
#             image_date = datetime.strptime(image_record['image_date'], "%Y-%m-%d").date()
#             image_time = datetime.strptime(image_record['image_time'], "%H:%M").time()
#             image_datetime = datetime.combine(image_date, image_time)

#             # Check if image_datetime is within the start and end datetime of the routine
#             if datetime_start <= image_datetime <= datetime_end:
#                 # Increment the medication_count for that routine
#                 datetime_combination['medication_count'] += 1

#     # Calculate overall statistics
#     total_medications_recorded = sum(datetime_combination['medication_count'] for datetime_combination in unique_datetime_list)
#     total_medications_required = sum(datetime_combination['total_medications_required'] for datetime_combination in unique_datetime_list)
#     missed_medications = total_medications_required - total_medications_recorded

#     # Prepare data for the line chart
#     weeks_data = {
#         'Week': [],
#         'Missed Medications': []
#     }

#     for datetime_combination in unique_datetime_list:
#         week_start = datetime_combination['start_date'] - timedelta(days=datetime_combination['start_date'].weekday())
#         week_label = f"{week_start.strftime('%Y-%m-%d')} to {(week_start + timedelta(days=6)).strftime('%Y-%m-%d')}"
#         weeks_data['Week'].append(week_label)
        
#         # Calculate missed medications for the current routine
#         missed_medications = datetime_combination['total_medications_required'] - datetime_combination['medication_count']
#         weeks_data['Missed Medications'].append(missed_medications)

#     # Create a line chart using Plotly
#     chart = px.line(x=weeks_data['Week'], y=weeks_data['Missed Medications'], labels={'x': 'Week', 'y': 'Missed Medications'})
#     chart.update_layout(title='Missed Medications Over Time (Weekly)')

#     # Convert Plotly chart to HTML
#     chart_html = plot(chart, output_type='div')

#     # Pass overall statistics and chart to the HTML template
#     return render_template('pill_detection/dashboard.html',
#                         total_medications_recorded=total_medications_recorded,
#                         total_medications_required=total_medications_required,
#                         missed_medications=missed_medications,
#                         chart_html=chart_html)