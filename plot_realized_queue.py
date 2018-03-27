import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from scipy import stats
import argparse
import datetime as dt
import sys
import fnmatch
import re


reportsdir = '/home/atd2data/reports/summary'
minversion = 0.6

cols = ['gufi', 'runway', 'target', 'actualOff', 'actualOut', 'Total Excess Taxi', \
'Ramp Excess Taxi', 'AMA Excess Taxi', 'AMA model', 'predictedDelay', 'apreq','edct', 'Gate Hold', 'Compliance']
runwayVec = ['18L', '18C', '36R', '36C'] # hardcoded to CLT

def plot_queue(targetdate, banknum):
	print(targetdate)
	dateVar = targetdate.strftime('%Y-%m-%d')
	print(dateVar)
	targetdir = os.path.join(reportsdir, '{:d}'.format(targetdate.year), '{:02d}'.format(targetdate.month), '{:02d}'.format(targetdate.day))
	targetout = os.path.join('data', '{:d}'.format(targetdate.year), '{:02d}'.format(targetdate.month), '{:02d}'.format(targetdate.day), 'bank{}'.format(banknum))

	# Load flightSummary, if able
	ffs_all = fnmatch.filter(os.listdir(targetdir), '*fullFlightSummary*')
	ffs_AAL = fnmatch.filter(os.listdir(targetdir), '*fullFlightSummary_AAL*')
	ffs = set(ffs_all) - set(ffs_AAL)
	if not ffs:
		print('No fullFlightSummary files found. Skipping day...')
		return
	versions = []
	for f in ffs:
		ver = re.compile('v\d+.\d+').findall(f)
		if len(ver) > 0:	
			vernum = float(ver[0].strip('v'))
			versions.append(vernum)
	if not versions:
		print('Version number for fullFlightSummary files not found; file to load is ambiguous. Skipping day...')
		return
	latestver = max(versions)
	# check version >= 0.6
	if latestver < minversion:
		print('fullFlightSummary version {} is less than {}. Skipping day...'.format(latestver, minversion))
		return
	for f in ffs:
		if fnmatch.fnmatch(f, '*fullFlightSummary.v{}*'.format(latestver)): 
			targetfile = f
	dfSummary = pd.read_csv(os.path.join(targetdir, targetfile), sep=',',index_col=False)

	df_queue = pd.read_csv(os.path.join(targetout, 'summary_{}_bank{}.csv'.format(dateVar, banknum)), sep=',',index_col=False)

	#####################

	stMF = targetdate.strftime('%Y%m%d')
	metered = True
	try:
		#dfMF = pd.read_csv('~/Documents/meteredFlights/metered_flights_' + stMF + '.csv'  , sep=',' , index_col=False)
		dfMF = pd.read_csv('metered_flights_{}.csv'.format(stMF), sep=',', index_col=False)
		#dfMF = pd.read_csv('~/Documents/MeteringAnalysis/Delay/data/bank2/bank2_MATM_data_2018-02-26.csv'  , sep=',' , index_col=False)
	except:
		print('metered flights output not found')
		metered = False

	#####################
	
	idx = -1
	dfPlot = pd.DataFrame(np.empty((1, len(cols)), dtype=object), columns=cols)
	for flight in range(len(dfSummary['gufi'])):
		if dfSummary['isDeparture'][flight] == True:
			if dfSummary['actual_off_bank_number'][flight] == banknum:
				idx += 1
				#excessRamp0 = (dfSummary['Actual_Ramp_Taxi_Out_Time'][flight] - dfSummary['Ramp_Taxi_Pred_at_Ramp_Taxi_Start'][flight]) / float(60)
				dfPlot.loc[idx,'gufi'] = dfSummary.loc[flight, 'gufi']
				dfPlot.loc[idx,'runway'] = dfSummary.loc[flight, 'departure_runway_position_derived']
				dfPlot.loc[idx,'Ramp Excess Taxi'] = max([0, dfSummary.loc[flight,'excess_departure_ramp_taxi_time']/float(60)])
				dfPlot.loc[idx,'AMA model'] = dfSummary.loc[flight, 'undelayed_departure_ama_transit_time']
				dfPlot.loc[idx,'AMA Excess Taxi'] = max([0, dfSummary.loc[flight, 'excess_departure_ama_taxi_time']/float(60)])
				dfPlot.loc[idx,'Total Excess Taxi'] = dfPlot.loc[idx, 'Ramp Excess Taxi'] +  dfPlot.loc[idx, 'AMA Excess Taxi']
				dfPlot.loc[idx,'actualOff'] = dfSummary.loc[flight, 'departure_runway_actual_time']
				dfPlot.loc[idx,'actualOut'] = dfSummary.loc[flight, 'departure_stand_actual_time']
								
				
				dfPlot.loc[idx, 'apreq'] = False
				dfPlot.loc[idx, 'edct'] = False
				
				if str(dfSummary.loc[flight, 'apreq_final']) != 'nan':
					dfPlot.loc[idx, 'apreq'] = True
				

				if str(dfSummary.loc[flight, 'edct_final']) != 'nan':
					dfPlot.loc[idx, 'edct'] = True
				

				if dfSummary.loc[flight, 'hold_indicator'] == True:
					dfPlot.loc[idx, 'Gate Hold'] = dfSummary.loc[flight, 'actual_gate_hold']


				dfCompliance = dfMF[ dfMF['gufi'] == dfSummary.loc[dfSummary.index[flight],'gufi']]
				if len(dfCompliance['gufi']) > 0:
					
					compliance = pd.Timedelta(pd.Timestamp(dfCompliance.loc[dfCompliance.index[0],'departure_stand_actual_time']) 
					- pd.Timestamp(dfCompliance.loc[dfCompliance.index[0],'departure_stand_surface_metered_time_value_ready'])
					).total_seconds() / float(60)
				
					if str(compliance) != 'nan':
						dfPlot.loc[idx,'Compliance'] = compliance

	dfPlot = dfPlot.sort_values(by=['actualOff'])
	dfPlot = dfPlot.reset_index(drop=True)
	for rwy in range(len(runwayVec)):
		for row in range(len(df_queue)):
			if df_queue.loc[row, 'runway'] == runwayVec[rwy]:
				targetVec = str(df_queue.loc[row, 'target']).split('--')
				targetTime = str(df_queue.loc[row, 'target_timestamp']).split('--')
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
		
		if len(dfSorted) > 0:

			dfSorted.to_csv(os.path.join(targetout, 'realized_queue_{}_bank{}_{}.csv'.format(dateVar, banknum, runwayVec[rwy]))) # why is this repeated?
			
			ax = dfSorted.plot(x='actualOff', y='target',figsize=(14,10))
			#dfSorted.plot.bar(x='actualOff', y=['Total Excess Taxi', 'AMA Excess Taxi','Ramp Excess Taxi'], color = ['cyan' , 'magenta' , 'grey'],alpha=0.6,ax=ax)
			dfSorted.plot.bar(x='actualOff', y=['AMA Excess Taxi','Ramp Excess Taxi'],width=0.3, position = -0.25, color = [ 'magenta' , 'grey'],alpha=0.6,ax=ax)
			dfSorted.plot.bar(x='actualOff', y=['Total Excess Taxi','Gate Hold'], width = 0.15, position = 0.5, stacked=True, color = [ 'cyan' , 'red'],alpha=0.6,ax=ax)
			dfSorted.plot.bar(x='actualOff', y=['Compliance'], width = 0.15, position = 1.5, color = [ 'blue'],alpha=0.6,ax=ax)
			plt.title('Runway ' + runwayVec[rwy] + ' ' + targetdate.strftime('%Y-%m-%d'))
			plt.ylabel('Excess Taxi Time [Minutes]')
			plt.xlabel('Actuall Off Time [UTC]')
			# minVal = 0
			# if dfSorted['Compliance'].min() < 0:
			# 	minVal = dfSorted['Compliance'].min()
			# if minVal < -20:
			# 	minVal = -20
			plt.ylim([-10,35])
			
			apreq_x = []
			apreq_y = []
			edct_x = []
			edct_y = []
			for i in range(len(dfSorted)):
				if dfSorted.loc[dfSorted.index[i],'apreq'] == True:
					apreq_x.append(i)
					apreq_y.append(dfSorted.loc[dfSorted.index[i],'target'])
				if dfSorted.loc[dfSorted.index[i],'edct'] == True:
					edct_x.append(i)
					edct_y.append(dfSorted.loc[dfSorted.index[i],'target'])
			
			plt.plot(apreq_x,apreq_y,'o',markersize = 10,color='orange',label='Apreq Flight')
			plt.plot(edct_x,edct_y,'o',markersize = 10,color='green',label='EDCT Flight')
			plt.legend(loc='upper left')

			plt.tight_layout()
			#plt.savefig('figs/flightSpecificDelay/' + runwayVec[rwy]+dateVecIADS[date] + '.png')
			plt.savefig(os.path.join(targetout, 'realized_queue_fig_{}_bank{}_{}.png'.format(dateVar, banknum, runwayVec[rwy])))
			plt.close('all')
			dfSorted.to_csv(os.path.join(targetout, 'realized_queue_{}_bank{}_{}.csv'.format(dateVar, banknum, runwayVec[rwy]))) # why is this repeated?



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
