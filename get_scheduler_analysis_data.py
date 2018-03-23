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
	flight_key, general_stream, msg_time, flight_status,schedule_priority, model_schedule_state,
	gate, gate_eta, gate_sta, spot, spot_eta, spot_sta,
	runway, runway_eta, runway_sta, (runway_sta - runway_eta) as ttot_minus_utot,
	metering_mode, metering_display, target_queue_buffer, metering_display_entry_offset, metering_display_exit_offset
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
	targetout = os.path.join('data', '{:d}'.format(targetdate.year), '{:02d}'.format(targetdate.month), '{:02d}'.format(targetdate.day), 'bank{}'.format(banknum))
	df.to_csv(os.path.join(targetout, 'scheduler_analysis_data_{}_bank{}.csv'.format(dayVar, banknum)))


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
