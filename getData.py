import psycopg2
import pandas.io.sql as psql
import numpy as np
import pandas as pd
import os
import argparse
import datetime as dt

DBHOST = 'localhost'
DBPORT = '5909'
DBNAME = 'fuserclt'
DBUSER = 'fuserclt'
DBPW = 'fuserclt'


#print('Attempting to connect to database')
##conn = psycopg2.connect("dbname='fuser' user='fuser' password='fuser' host='localhost'  ")
#conn = psycopg2.connect("dbname='fuserclt' user='fuserclt' password='fuserclt' host='localhost' port='5909' ")

#bank2 = False
#bank3 = True

#dayVarVec = []

#daySt = '2017-12-'
#for i in range(3,32):
#	if i < 10:
#		dayVarVec.append(daySt + '0' + str(i))
#	else:
#		dayVarVec.append(daySt + str(i))


def fGetData(targetdate, banknum, conn):
	dayVar = targetdate.strftime('%Y-%m-%d')
	if pd.Timestamp(dayVar + ' 00:00:00') <= pd.Timestamp('2018-03-10 00:00:00'): # TODO extract and handle daylight savings for all years
		if banknum == 2:
			time1 = dayVar + ' 14:00:00'
			time2 = dayVar + ' 16:00:00'
			print(time1)
		if banknum == 3:
			time1 = dayVar + ' 16:00:00'
			time2 = dayVar + ' 18:00:00'
			print(time1)
	else:
		if banknum == 2:
			time1 = dayVar + ' 13:00:00'
			time2 = dayVar + ' 15:00:00'
			print(time1)
		if banknum == 3:
			time1 = dayVar + ' 15:00:00'
			time2 = dayVar + ' 17:00:00'
			print(time1)

	q = '''SELECT
	tf.flight_key, tf.general_stream, tfs.schedule_priority, tfs.model_schedule_state, 
	trun.runway, tg.gate, smd.eta_msg_time, rt.scheduled_time, rt.estimated_time, (rt.scheduled_time - rt.estimated_time) as ttot_minus_utot,
	tfix.fix, smd.metering_display, smd.metering_mode, rt.schedule_sequence, smd.target_queue_buffer, smd.metering_display_entry_offset,
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

#	df = psql.read_sql(q, conn)

#	#df.to_csv('DebugMeteringFuserWarehouseV2' + str(callsign) + '.csv')
#	if bank2:
#		df.to_csv('data/bank2/bank2_all_data_' + dayVar + '.csv')
#	if bank3:
#		df.to_csv('data/bank3/bank3_all_data_' + dayVar + '.csv')

	df = psql.read_sql(q, conn)
	targetout = os.path.join('data', '{:d}'.format(targetdate.year), '{:02d}'.format(targetdate.month), '{:02d}'.format(targetdate.day), 'bank{}'.format(banknum))
	df.to_csv(os.path.join(targetout, 'scheduler_analysis_data_{}_bank{}.csv'.format(dayVar, banknum)))


#for dayVar in dayVarVec:
#	fGetData(dayVar,conn)



def run(start_date, number_of_days, bank):
	print('Attempting to connect to database')
	conn = psycopg2.connect(dbname=DBNAME, user=DBUSER, password=DBPW, host=DBHOST, port=DBPORT)
	for day in range(number_of_days):
		day_start = start_date + dt.timedelta(days = day)
		targetout = os.path.join('data','{:d}'.format(day_start.year), '{:02d}'.format(day_start.month), '{:02d}'.format(day_start.day), 'bank{}'.format(bank))
		if not os.path.exists(targetout):
			os.makedirs(targetout)
		fGetData(day_start, bank, conn)
	conn.close()


if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('startdate', help='Start date of analysis (YYYYMMDD)')
	parser.add_argument('-d', '--days', help='Number of days to run analysis, default 1', 
			type=int, default=1)
	parser.add_argument('-b', '--bank', help='Bank number to run analysis on, default 2', 
			type=int, default=2)

	args = parser.parse_args()
    
	start_date = dt.datetime.strptime(args.startdate, '%Y%m%d').date()

	run(start_date, args.days, args.bank)
