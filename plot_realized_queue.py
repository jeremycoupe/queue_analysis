import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from scipy import stats


cols = ['gufi','runway','target','actualOff', 'actualOut' , 'Total Excess Taxi' , 'Ramp Excess Taxi' , 'AMA Excess Taxi' ,'AMA model', 'predictedDelay','controlled' ,'Gate Hold']
#dfPlot = pd.DataFrame(np.empty((1,len(cols)), dtype=object),columns=cols)

runwayVec = ['18L' , '18C' , '36R' , '36C']
files = os.listdir('/Users/wcoupe/Documents/tmp')

# dfMeter = pd.read_csv('~/Desktop/surface_metering_summary_bank2.csv', sep=',',index_col=False)
 

def getMeterDateString(date_variable):
	yearSt = date_variable[2:4]
	monthSt = date_variable[4:6]
	daySt = date_variable[6:8]
	if monthSt[0] == '0':
		stMeterSummary = monthSt[1] + '/'
	else:
		stMeterSummary = monthSt + '/'
	
	if daySt[0] == '0':
		stMeterSummary = stMeterSummary + daySt[1] + '/' + yearSt
	else:
		stMeterSummary = stMeterSummary + daySt + '/' + yearSt

	return stMeterSummary


for fileName in range(len(files)):
	if 'KCLT.fullFlightSummary.v0.6' in files[fileName]:
		dfSummary = pd.read_csv('/Users/wcoupe/Documents/tmp/' + files[fileName], sep=',',index_col=False) # what time does this get sync'd to ames?
		dateVar = str(files[fileName]).split('.')[4]
		print(dateVar)

		df_queue = pd.read_csv('/Users/wcoupe/Documents/queue_analysis/data/2018/03/16/bank2/summary_2018-03-16_bank2.csv',sep=',',index_col=False)



		st_date_var = getMeterDateString(dateVar)
			
		# target = False																			
		# for row in range(len(dfMeter['date'])):
		# 	if dfMeter.loc[row,'date'] == st_date_var:
				
		# 		target=dfMeter.loc[row,'time_based_target_excess_queue_minutes']


		
		idx = -1

		dfPlot = pd.DataFrame(np.empty((1,len(cols)), dtype=object),columns=cols)
		for flight in range(len(dfSummary['gufi'])):
			if dfSummary['isDeparture'][flight] == True:
				if str(dfSummary['actual_off_bank_number'][flight]) == str(2.0):
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


				dfSorted.to_csv('/Users/wcoupe/Desktop/' + dateVar + '_'+ runwayVec[rwy]+ '.csv')
				
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
				plt.savefig('/Users/wcoupe/Desktop/' + dateVar + '_'+ runwayVec[rwy]+ '.png')
				plt.close('all')
				dfSorted.to_csv('/Users/wcoupe/Desktop/' + dateVar + '_'+ runwayVec[rwy]+ '.csv')





