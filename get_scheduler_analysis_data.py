import psycopg2
import pandas.io.sql as psql
import numpy as np
import pandas as pd
import os

print('Attempting to connect to database')
#conn = psycopg2.connect("dbname='fuser' user='fuser' password='fuser' host='localhost'  ")
conn = psycopg2.connect("dbname='fuserclt' user='fuserclt' password='fuserclt' host='localhost'  ")

bank2 = True
bank3 = False

dayVarVec = []

daySt = '2018-03-'
for i in range(1,17):
	if i < 10:
		dayVarVec.append(daySt + '0' + str(i))
	else:
		dayVarVec.append(daySt + str(i))


def fGetData(dayVar,conn):
	
	if pd.Timestamp(dayVar + ' 00:00:00') <= pd.Timestamp('2018-03-10 00:00:00'):

		if bank2:
			time1 = dayVar + ' 14:00:00'
			time2 = dayVar + ' 16:00:00'
			print(time1)
		if bank3:
			time1 = dayVar + ' 16:00:00'
			time2 = dayVar + ' 18:00:00'
			print(time1)
	else:
		if bank2:
			time1 = dayVar + ' 13:00:00'
			time2 = dayVar + ' 15:00:00'
			print(time1)
		if bank3:
			time1 = dayVar + ' 15:00:00'
			time2 = dayVar + ' 17:00:00'
			print(time1)

	q = '''SELECT
	flight_key,general_stream,msg_time,flight_status,schedule_priority,model_schedule_state,
	gate,gate_eta,gate_sta,spot,spot_eta,spot_sta,
	runway,runway_eta,runway_sta,(runway_sta - runway_eta) as ttot_minus_utot,
	metering_mode,metering_display,target_queue_buffer,metering_display_entry_offset,metering_display_exit_offset
	FROM
	scheduler_analysis
	WHERE
	msg_time > '%s'
	and msg_time < '%s'
	order by 
	msg_time ASC,
	runway
	'''%(time1,time2)


	df = psql.read_sql(q, conn)

	#df.to_csv('DebugMeteringFuserWarehouseV2' + str(callsign) + '.csv')
	if bank2:
		df.to_csv('data/bank2/scheduler_analysis/bank2_scheduler_analysis_data_' + dayVar + '.csv')
	if bank3:
		df.to_csv('data/bank3/scheduler_analysis/bank3_scheduler_analysis_data_' + dayVar + '.csv')


for dayVar in dayVarVec:
	fGetData(dayVar,conn)
