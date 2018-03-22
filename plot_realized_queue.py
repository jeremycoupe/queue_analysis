import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from scipy import stats
import argparse
import datetime as dt
import sys
import fnmatch


reportsdir = '/home/atd2data/reports/summary'

cols = ['gufi','runway','target','actualOff', 'actualOut' , 'Total Excess Taxi' , 'Ramp Excess Taxi' , 'AMA Excess Taxi' ,'AMA model', 'predictedDelay','controlled' ,'Gate Hold']
runwayVec = ['18L' , '18C' , '36R' , '36C'] # hardcoded to CLT

def plot_queue(targetdate, banknum):
	targetdir = os.path.join(reportsdir, '{:d}'.format(targetdate.year), '{:02d}'.format(targetdate.month), '{:02d}'.format(targetdate.day))
	targetdate_dir = targetdate.strftime('%Y/%m/%d') #TODO clean up with os.path
	try:
		for f in os.listdir(targetdir):
			if fnmatch.fnmatch(f, '*fullFlightSummary.v*'):
				targetfile = f
		dfSummary = pd.read_csv(os.path.join(targetdir, targetfile), sep=',',index_col=False)
	except:
		print('file at {} does not exist'.format(os.path.join(targetdir, targetfile)))
		sys.exit()
	df_queue = pd.read_csv('data/{0}/bank{1}/summary_{2}_bank{1}.csv'.format(targetdate.strftime('%Y/%m/%d'), banknum, targetdate.strftime('%Y-%m-%d')),sep=',',index_col=False)

	dateVar = targetdate.strftime('%Y%m%d')
	print(dateVar)
	
	idx = -1
	dfPlot = pd.DataFrame(np.empty((1,len(cols)), dtype=object),columns=cols)
	for flight in range(len(dfSummary['gufi'])):
		if dfSummary['isDeparture'][flight] == True:
			if dfSummary['actual_off_bank_number'][flight] == banknum: # why convert to str?
				idx+=1
				#excessRamp0 = (dfSummary['Actual_Ramp_Taxi_Out_Time'][flight] - dfSummary['Ramp_Taxi_Pred_at_Ramp_Taxi_Start'][flight]) / float(60)
				dfPlot.loc[idx,'gufi'] = dfSummary.loc[flight,'gufi']
				dfPlot.loc[idx,'runway'] = dfSummary.loc[flight,'departure_runway_position_derived']
				dfPlot.loc[idx,'Ramp Excess Taxi'] = max([0,dfSummary.loc[flight,'excess_departure_ramp_taxi_time']/float(60)])
				dfPlot.loc[idx,'AMA model'] = dfSummary.loc[flight,'undelayed_departure_ama_transit_time']
				dfPlot.loc[idx,'AMA Excess Taxi'] = max([0,dfSummary.loc[flight,'excess_departure_ama_taxi_time']/float(60)])
				dfPlot.loc[idx,'Total Excess Taxi'] = dfPlot.loc[idx,'Ramp Excess Taxi'] +  dfPlot.loc[idx,'AMA Excess Taxi']
				dfPlot.loc[idx,'actualOff'] = dfSummary.loc[flight,'departure_runway_actual_time']
				dfPlot.loc[idx,'actualOut'] = dfSummary.loc[flight,'departure_stand_actual_time']
								
				if str(dfSummary.loc[flight,'apreq_final']) != 'nan':
					dfPlot.loc[idx,'controlled'] = True
				else:
					dfPlot.loc[idx,'controlled'] = False

				if dfSummary.loc[flight,'hold_indicator'] == True:
					dfPlot.loc[idx,'Gate Hold'] = dfSummary.loc[flight,'actual_gate_hold']

	dfPlot = dfPlot.sort_values(by=['actualOff'])
	dfPlot = dfPlot.reset_index(drop=True)
	for rwy in range(len(runwayVec)):
		for row in range(len(df_queue)):
			if df_queue.loc[row,'runway'] == runwayVec[rwy]:
				targetVec = str(df_queue.loc[row,'target']).split('--')
				targetTime = str(df_queue.loc[row,'target_timestamp']).split('--')
				# print(targetVec)
				# print(targetTime)
		
				if len(targetVec) > 1:
					count=0
					for flight in range(len(dfPlot)):
						if dfPlot.loc[flight,'runway'] == runwayVec[rwy]:
							# print(pd.Timestamp(dfPlot.loc[flight,'actualOff']))
							# print(pd.Timestamp(targetTime[count]))
							# print(pd.Timedelta(pd.Timestamp(dfPlot.loc[flight,'actualOff']) - pd.Timestamp(targetTime[count]) ).total_seconds())
							# print(count)
							if count < len(targetVec) -1 :
								if pd.Timedelta(pd.Timestamp(dfPlot.loc[flight,'actualOff']) - pd.Timestamp(targetTime[count+1]) ).total_seconds() < 0:
									dfPlot.loc[flight,'target'] = float(targetVec[count])
								else:
									#print('HERE')
									count+=1
									dfPlot.loc[flight,'target'] = float(targetVec[count])
							else:
								dfPlot.loc[flight,'target'] = float(targetVec[count])
				else:
					for flight in range(len(dfPlot)):
						if dfPlot.loc[flight,'runway'] == runwayVec[rwy]:
							dfPlot.loc[flight,'target'] = float(targetVec[0])

	for rwy in range(len(runwayVec)):
		df = dfPlot[ dfPlot['runway'] == runwayVec[rwy]]
		#print(df)
		dfSorted = df.sort_values(by=['actualOff'])
		dfSorted = dfSorted.reset_index(drop=True)
		
		if len(dfSorted)>0:

			dfSorted.to_csv('data/{0}/bank{1}/{2}_bank{1}_{3}.csv'.format(targetdate_dir, banknum, dateVar, runwayVec[rwy]))
			
			ax = dfSorted.plot(x='actualOff',y='target',figsize=(14,10))
			#dfSorted.plot.bar(x='actualOff',y=['Total Excess Taxi', 'AMA Excess Taxi','Ramp Excess Taxi'], color = ['cyan' , 'magenta' , 'grey'],alpha=0.6,ax=ax)
			dfSorted.plot.bar(x='actualOff',y=['AMA Excess Taxi','Ramp Excess Taxi'],width=0.3, position = -0.25, color = [ 'magenta' , 'grey'],alpha=0.6,ax=ax)
			dfSorted.plot.bar(x='actualOff',y=['Total Excess Taxi','Gate Hold'], width = 0.15, position = 0.5, stacked=True, color = [ 'cyan' , 'red'],alpha=0.6,ax=ax)
			plt.title('Runway ' + runwayVec[rwy] + ' ' + dateVar)
			plt.ylabel('Excess Taxi Time [Minutes]')
			plt.xlabel('Actuall Off Time [UTC]')
			plt.ylim([0,35])
			plt.tight_layout()
			#plt.savefig('figs/flightSpecificDelay/' + runwayVec[rwy]+dateVecIADS[date] + '.png')
			plt.savefig('data/{0}/bank{1}/{2}_bank{1}_{3}.png'.format(targetdate_dir, banknum, dateVar, runwayVec[rwy]))
			plt.close('all')
			dfSorted.to_csv('data/{0}/bank{1}/{2}_bank{1}_{3}.csv'.format(targetdate_dir, banknum, dateVar, runwayVec[rwy])) # why is this repeated?


def run(start_date, number_of_days, bank):
	for day in range(number_of_days):
		day_start = start_date + dt.timedelta(days = day)
		plot_queue(day_start, bank)


if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('startdate', help = 'Start date of analysis (YYYYMMDD)')
	parser.add_argument('-d', '--days', help='Number of days to run analysis, default 1', 
			type = int, default = 1)
	parser.add_argument('-b', '--bank', help='Bank number to run analysis on, default 2', 
			type = int, default = 2)

	args = parser.parse_args()
    
	start_date = dt.datetime.strptime(args.startdate, '%Y%m%d').date()

	run(start_date, args.days, args.bank)
