import psycopg2
import pandas.io.sql as psql
import numpy as np
import pandas as pd
import os

print('Attempting to connect to database')
#conn = psycopg2.connect("dbname='fuser' user='fuser' password='fuser' host='localhost'  ")
conn = psycopg2.connect("dbname='fuserclt' user='fuserclt' password='fuserclt' host='localhost' port='5909' ")

bank2 = False
bank3 = True

dayVarVec = []

daySt = '2018-02-'
for i in range(19,29):
	if i < 10:
		dayVarVec.append(daySt + '0' + str(i))
	else:
		dayVarVec.append(daySt + str(i))


# for i in range(19,23):
# 	if i < 10:
# 		dayVarVec.append(daySt + '0' + str(i))
# 	else:
# 		dayVarVec.append(daySt + str(i))

# daySt = '2018-01-'
# for i in range(9,32):
# 	if i < 10:
# 		dayVarVec.append(daySt + '0' + str(i))
# 	else:
# 		dayVarVec.append(daySt + str(i))



def fGetData(dayVar,conn):
	if bank2:
		time1 = dayVar + ' 13:30:00'
		time2 = dayVar + ' 16:00:00'
		print(time1)
	if bank3:
		time1 = dayVar + ' 15:30:00'
		time2 = dayVar + ' 18:00:00'
		print(time1)

	q = '''SELECT
	tf.flight_key, tf.general_stream, tfs.schedule_priority, tfs.model_schedule_state, 
	trun.runway, tg.gate, smd.eta_msg_time, rt.scheduled_time, rt.estimated_time, (rt.scheduled_time - rt.estimated_time) as ttot_minus_utot,
	tfix.fix,smd.metering_display,smd.metering_mode, rt.schedule_sequence, smd.target_queue_buffer, smd.metering_display_entry_offset,
	smd.metering_display_exit_offset, tad.ac_type, tad.weight_class
	FROM
	tactical_route_times rt
	INNER JOIN tactical_flight tf ON rt.tactical_flight_id = tf.id
	INNER JOIN tactical_fix tfix ON rt.tactical_fix_schedule_id = tfix.id
	INNER JOIN tactical_route_info ri ON rt.tactical_route_info_id = ri.id
	INNER JOIN tactical_runway trun ON ri.tactical_runway_id = trun.id
	INNER JOIN tactical_schedule_meta_data smd ON rt.tactical_schedule_meta_data_id = smd.id
	INNER JOIN tactical_flight_state tfs on rt.tactical_flight_state_id = tfs.id
	INNER JOIN tactical_gate tg on ri.tactical_gate_id = tg.id
	INNER JOIN tactical_aircraft_data tad on rt.tactical_aircraft_data_id = tad.id
	WHERE
	smd.eta_msg_time > '%s'
	and smd.eta_msg_time < '%s'
	-- and tfix.fix = tg.gate
	order by smd.eta_msg_time, trun.runway, rt.estimated_time
	'''%(time1,time2)
	# -- tf.flight_key = '%s'
	# -- and smd.eta_msg_time > '2017-12-13 00:00:00'
	# -- and tf.general_stream = 'DEPARTURE'
	# -- and tf.general_stream = 'DEPARTURE'
	#%(callsign)

	df = psql.read_sql(q, conn)

	#df.to_csv('DebugMeteringFuserWarehouseV2' + str(callsign) + '.csv')
	if bank2:
		df.to_csv('data/bank2/bank2_all_data_' + dayVar + '.csv')
	if bank3:
		df.to_csv('data/bank3/bank3_all_data_' + dayVar + '.csv')


for dayVar in dayVarVec:
	fGetData(dayVar,conn)
