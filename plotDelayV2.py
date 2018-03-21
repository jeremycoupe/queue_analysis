import numpy as np
import pandas as pd
import psycopg2
import pandas.io.sql as psql
import matplotlib.pyplot as plt

randomColors = np.load('/home/milin/analysis/queue_analysis/randomColors.npy')

# TODO: make this work for new scheduler tables

bank2 = True
bank3 = False


alphaVal = 0.6

dateVec = []

#daySt = '2018-02-'

# for i in range(26,27):
# 	if i < 10:
# 		dateVec.append(daySt + '0' + str(i))
# 	else:
# 		dateVec.append(daySt + str(i))

# daySt = '2018-01-'
# for i in range(19,32):
# 	if i < 10:
# 		dateVec.append(daySt + '0' + str(i))
# 	else:
# 		dateVec.append(daySt + str(i))


daySt = '2017-12-'
for i in range(3,32):
	if i < 10:
		dateVec.append(daySt + '0' + str(i))
	else:
		dateVec.append(daySt + str(i))

totalNumberFlutter = 0
debug_except_notRwySw = pd.DataFrame()
for date in range(len(dateVec)):
	print(dateVec[date])
	if bank2:
		file = 'data/bank2/bank2_all_data_' + dateVec[date] + '.csv'
	if bank3:
		file = 'data/bank3/bank3_all_data_' + dateVec[date] + '.csv'
	
	df = pd.read_csv(file)

	df = df[ (df['general_stream'] == 'DEPARTURE') ]
	runwayVec = df['runway'].unique()

	stMF = dateVec[date].replace('-','')
	metered = True
	try:
		#dfMF = pd.read_csv('~/Documents/meteredFlights/metered_flights_' + stMF + '.csv'  , sep=',' , index_col=False)
		dfMF = pd.read_csv('/home/milin/analysis/queue_analysis/metered_flights_' + stMF + '.csv'  , sep=',' , index_col=False)
		#dfMF = pd.read_csv('~/Documents/MeteringAnalysis/Delay/data/bank2/bank2_MATM_data_2018-02-26.csv'  , sep=',' , index_col=False)
	except:
		try:
			dfMF = pd.read_csv('~/Documents/MeteringAnalysis/Delay/data/bank2/bank2_MATM_data_'+ dateVec[date] + '.csv'  , sep=',' , index_col=False)
		except:
			metered = False

	cols0 = ['date','runway','baloon_metric','average_compliance','count_non_compliant_5',\
	'meter_switch_on_off','meter_on','meter_off','count_apreq','count_edct','count_exempt','count_runway_switch',\
	'count_planned' , 'count_ready' , 'count_uncertain','count_ga_uncertain','count_ga_apreq',\
	'count_ga_edct']
	df_summary = pd.DataFrame(np.empty((1,len(cols0)), dtype=object),columns=cols0)
	idS = -1

	for rwy in range(len(runwayVec)):
# TODO: instead of filter by TIME_BASED_METERING, use bank start and end
		dfCurrentRunway = df[ (df['metering_mode'] == 'TIME_BASED_METERING') & (df['runway'] == runwayVec[rwy]) \
		& (df['fix'] == df['runway']) & (df['general_stream'] == 'DEPARTURE') ]
		
		if len(dfCurrentRunway)>0:
			dfCurrentRunway = dfCurrentRunway.sort_values(by=['eta_msg_time','scheduled_time'])
			dfCurrentRunway.to_csv('data/bank2/debug/debug'+runwayVec[rwy]+dateVec[date]+'.csv')

			activeVec = []

			cols = ['gufi','runway','ts','eta_msg_time','compliance','previous_state']

			df_compliance = pd.DataFrame(np.empty((1,len(cols)), dtype=object),columns=cols)
			idx = -1
			
			etaMsgVec0 = dfCurrentRunway['eta_msg_time'].drop_duplicates()
			etaMsgVec = etaMsgVec0.reset_index(drop=True)

			displayEta = []
			# print(len(etaMsgVec))
			for i in range(len(etaMsgVec)):
				if i % 90 == 0:
					displayEta.append(str(etaMsgVec[i]).split('.')[0])
					#print('HERE')

			#print(etaMsgVec)

			maxActive = np.zeros(len(etaMsgVec))
			maxPlanned = np.zeros(len(etaMsgVec))
			maxReady = np.zeros(len(etaMsgVec))
			maxUncertain = np.zeros(len(etaMsgVec))
			meanActive = np.zeros(len(etaMsgVec))
			meanPlannedDelay = np.zeros(len(etaMsgVec))
			minPlannedDelay = np.zeros(len(etaMsgVec))
			numPlanned = np.zeros(len(etaMsgVec))
			numAboveTarget = np.zeros(len(etaMsgVec))
			numActive = np.zeros(len(etaMsgVec))
			numAMA = np.zeros(len(etaMsgVec))
			numReady = np.zeros(len(etaMsgVec))
			meterVec = np.zeros(len(etaMsgVec))
			complianceVec = np.zeros(len(etaMsgVec))
			exemptVec = np.zeros(len(etaMsgVec))
			balloonMetricVec = np.zeros(len(etaMsgVec))
			targetVec = np.zeros(len(etaMsgVec))
			upperBoundVec = np.zeros(len(etaMsgVec))
			lowerBoundVec = np.zeros(len(etaMsgVec))

			plt.figure(figsize = (14,10))

			for ts in range(len(etaMsgVec)):

				dfActive = dfCurrentRunway[ (dfCurrentRunway['eta_msg_time'] == etaMsgVec[ts] ) & (dfCurrentRunway['schedule_priority'].isin(['NO_PRIORITY_DEPARTURE_TAXI' , 'NO_PRIORITY_DEPARTURE_TAXI_AMA' ]))]
				dfAMA = dfCurrentRunway[ (dfCurrentRunway['eta_msg_time'] == etaMsgVec[ts] ) & (dfCurrentRunway['schedule_priority'] == 'NO_PRIORITY_DEPARTURE_TAXI_AMA')]
				dfGate = dfCurrentRunway[ (dfCurrentRunway['eta_msg_time'] == etaMsgVec[ts]) & (dfCurrentRunway['schedule_priority'] == 'GATE_DEPARTURE_PLANNED')]
				dfUncertain = dfCurrentRunway[ (dfCurrentRunway['eta_msg_time'] == etaMsgVec[ts]) & (dfCurrentRunway['schedule_priority'] == 'GATE_DEPARTURE_UNCERTAIN')]
				dfReady = dfCurrentRunway[ (dfCurrentRunway['eta_msg_time'] == etaMsgVec[ts]) & (dfCurrentRunway['schedule_priority'] == 'GATE_DEPARTURE_READY')]
				#print(pd.Timedelta(dfUncertain['ttot_minus_utot'].max())/np.timedelta64(1, 's'))
				
				dfActiveALL = dfCurrentRunway[ (dfCurrentRunway['eta_msg_time'] == etaMsgVec[ts] )\
				& (dfCurrentRunway['model_schedule_state'].isin(['OUT' , 'TAXI' , 'QUEUE']))]

				maxActive[ts] = pd.Timedelta(dfActive['ttot_minus_utot'].max()) / np.timedelta64(1, 's')
				maxPlanned[ts] = pd.Timedelta(dfGate['ttot_minus_utot'].max() ) /np.timedelta64(1, 's')
				maxUncertain[ts] = pd.Timedelta(dfUncertain['ttot_minus_utot'].max()) / np.timedelta64(1, 's')
				maxReady[ts] = pd.Timedelta(dfReady['ttot_minus_utot'].max()) / np.timedelta64(1, 's')
				meanActive[ts] = np.mean( pd.to_timedelta(dfGate['ttot_minus_utot']) / np.timedelta64(1, 's') )
				meanPlannedDelay[ts] =  np.mean( pd.to_timedelta(dfGate['ttot_minus_utot']) / np.timedelta64(1, 's') )
				numPlanned[ts] = len(np.array(pd.to_timedelta(dfGate['ttot_minus_utot'])))
				numActive[ts] = len(np.array(pd.to_timedelta(dfActive['ttot_minus_utot'])))
				numAMA[ts] = len(np.array(pd.to_timedelta(dfAMA['ttot_minus_utot'])))
				numReady[ts] = len(np.array(pd.to_timedelta(dfReady['ttot_minus_utot'])))
				dfMeter = dfCurrentRunway[ (dfCurrentRunway['eta_msg_time'] == etaMsgVec[ts] )]
				stRunway = str(dfMeter.loc[dfMeter.index[0],'metering_display']).split(',')
				
				target_df = dfCurrentRunway[ (dfCurrentRunway['eta_msg_time'] == etaMsgVec[ts] ) ]
				targetVec[ts] = target_df.loc[target_df.index[0],'target_queue_buffer'] / float(60000)
				upperBoundVec[ts] = target_df.loc[target_df.index[0],'metering_display_entry_offset'] / float(60000)
				lowerBoundVec[ts] = target_df.loc[target_df.index[0],'metering_display_exit_offset'] / float(60000)
				balloonMetricVec[ts] = max([0, maxActive[ts]/float(60) - targetVec[ts]] )
				#print(stRunway)
				if runwayVec[rwy] in stRunway:
					meterVec[ts] = 1

				
				for flight in range(len(dfActiveALL['flight_key'])):
					if dfActiveALL.loc[dfActiveALL.index[flight],'flight_key'] not in activeVec:
						activeVec.append(dfActiveALL.loc[dfActiveALL.index[flight],'flight_key'])
						if metered:
							if meterVec[ts] == 1:
								idx+=1
								

									
								dfLastSchedule = dfCurrentRunway[ (dfCurrentRunway['eta_msg_time'] == etaMsgVec[ts-1] ) \
								& (dfCurrentRunway['flight_key'] == dfActiveALL.loc[dfActiveALL.index[flight],'flight_key'] )]

								if len(dfLastSchedule) > 0:
									lastState = dfLastSchedule.loc[dfLastSchedule.index[0],'model_schedule_state']
									lastPriority = dfLastSchedule.loc[dfLastSchedule.index[0],'schedule_priority']
									lastGate = dfLastSchedule.loc[dfLastSchedule.index[0],'gate']
									

# TODO make identification of GAs more robust
									if 'GA' not in lastGate:
										if lastState in ['PUSHBACK_PLANNED','PUSHBACK_READY','PUSHBACK_UNCERTAIN']:
											df_compliance.loc[idx,'runway'] = runwayVec[rwy]
											df_compliance.loc[idx,'gufi'] = dfActiveALL.loc[dfActiveALL.index[flight],'flight_key']
											df_compliance.loc[idx,'ts'] = ts
											df_compliance.loc[idx,'eta_msg_time'] = etaMsgVec[ts]
											
											dfCompliance = dfMF[ dfMF['gufi'] == dfActiveALL.loc[dfActiveALL.index[flight],'flight_key'] ]
											if len(dfCompliance['gufi']) > 0:
												compliance = pd.Timedelta(pd.Timestamp(dfCompliance.loc[dfCompliance.index[0],'departure_stand_actual_time']) \
												- pd.Timestamp(dfCompliance.loc[dfCompliance.index[0],'departure_stand_surface_metered_time_value_ready']) ).total_seconds() / float(60)
											else:
												compliance = 'nan'

											df_compliance.loc[idx,'compliance'] = compliance
											df_compliance.loc[idx,'previous_state'] = lastState


										if lastPriority in ['APREQ_DEPARTURE','EDCT_DEPARTURE']:
											df_compliance.loc[idx,'runway'] = runwayVec[rwy]
											df_compliance.loc[idx,'gufi'] = dfActiveALL.loc[dfActiveALL.index[flight],'flight_key']
											df_compliance.loc[idx,'ts'] = ts
											df_compliance.loc[idx,'eta_msg_time'] = etaMsgVec[ts]
											#####
											lastTOBT = df[ (df['flight_key'] == dfActiveALL.loc[dfActiveALL.index[flight],'flight_key'] ) \
											& (df['fix'] == df['gate'] ) & (df['eta_msg_time'] == etaMsgVec[ts-1] ) ]
											if len(lastTOBT)>0:
												apreqCompliance = pd.Timedelta( pd.Timestamp(etaMsgVec[ts]) - \
												pd.Timestamp(lastTOBT.loc[lastTOBT.index[0],'scheduled_time']) ).total_seconds() / float(60)
												# print('APREQ COMPLIANCE')
												# print(apreqCompliance)
												df_compliance.loc[idx,'compliance'] = apreqCompliance
											#####
											df_compliance.loc[idx,'previous_state'] = lastPriority

										if lastPriority in ['EXEMPT_DEPARTURE']:
											df_compliance.loc[idx,'runway'] = runwayVec[rwy]
											df_compliance.loc[idx,'gufi'] = dfActiveALL.loc[dfActiveALL.index[flight],'flight_key']
											df_compliance.loc[idx,'ts'] = ts
											df_compliance.loc[idx,'eta_msg_time'] = etaMsgVec[ts]
											#####
											lastTOBT = df[ (df['flight_key'] == dfActiveALL.loc[dfActiveALL.index[flight],'flight_key'] ) \
											& (df['fix'] == df['gate'] ) & (df['eta_msg_time'] == etaMsgVec[ts-1] ) ]
											if len(lastTOBT)>0:
												exemptCompliance = pd.Timedelta( pd.Timestamp(etaMsgVec[ts]) - \
												pd.Timestamp(lastTOBT.loc[lastTOBT.index[0],'scheduled_time']) ).total_seconds() / float(60)
												# print('APREQ COMPLIANCE')
												# print(apreqCompliance)
												df_compliance.loc[idx,'compliance'] = exemptCompliance
											#####
											df_compliance.loc[idx,'previous_state'] = lastPriority
											# print('EXEMPT DEPARTURE FOUND')

									else:
										df_compliance.loc[idx,'runway'] = runwayVec[rwy]
										df_compliance.loc[idx,'gufi'] = dfActiveALL.loc[dfActiveALL.index[flight],'flight_key']
										df_compliance.loc[idx,'ts'] = ts
										df_compliance.loc[idx,'eta_msg_time'] = etaMsgVec[ts]
										#####
										lastTOBT = df[ (df['flight_key'] == dfActiveALL.loc[dfActiveALL.index[flight],'flight_key'] ) \
										& (df['fix'] == df['gate'] ) & (df['eta_msg_time'] == etaMsgVec[ts-1] ) ]
										if len(lastTOBT)>0:
											gaCompliance = pd.Timedelta( pd.Timestamp(etaMsgVec[ts]) - \
											pd.Timestamp(lastTOBT.loc[lastTOBT.index[0],'scheduled_time']) ).total_seconds() / float(60)
											# print('APREQ COMPLIANCE')
											# print(apreqCompliance)
											df_compliance.loc[idx,'compliance'] = gaCompliance
										#####
										if lastState in ['PUSHBACK_PLANNED','PUSHBACK_READY','PUSHBACK_UNCERTAIN']:
											df_compliance.loc[idx,'previous_state'] = 'GA ' + lastState
										if lastPriority in ['APREQ_DEPARTURE','EDCT_DEPARTURE','EXEMPT_DEPARTURE']:
											df_compliance.loc[idx,'previous_state'] = 'GA ' + lastPriority
								
								else:
									if ts > 0:
										print('LOOK INTO THIS POSSIBLE RUNWAY SWITCH')
										print(etaMsgVec[ts])
										print(runwayVec[rwy])
										print(dfActiveALL.loc[dfActiveALL.index[flight],'flight_key'])

										df_compliance.loc[idx,'runway'] = runwayVec[rwy]
										df_compliance.loc[idx,'gufi'] = dfActiveALL.loc[dfActiveALL.index[flight],'flight_key']
										df_compliance.loc[idx,'ts'] = ts
										df_compliance.loc[idx,'eta_msg_time'] = etaMsgVec[ts]
										df_compliance.loc[idx,'compliance'] = 0
										df_compliance.loc[idx,'previous_state'] = 'RUNWAY_SWITCH'
										tempDF = df[(df['fix'] == df['runway'])&(df['eta_msg_time'] == etaMsgVec[ts-1] )\
										&(df['flight_key'] == dfActiveALL.loc[dfActiveALL.index[flight],'flight_key'] ) ]
										previousRunway = tempDF.loc[tempDF.index[0],'runway']

										print('previousRunway is {}, runwayVec[rwy] is {}'.format(previousRunway, runwayVec[rwy]))
										if previousRunway != runwayVec[rwy]:
											print('CONFIRMED RUNWAY SWITCH')
										else: 
											print('Not a runway switch! Adding to file:') #TODO: write out to file
											debug_except_notRwySw = debug_except_notRwySw.append(pd.DataFrame(
												{'gufi':dfActiveALL.loc[dfActiveALL.index[flight],'flight_key'], 
												'rwy':runwayVec[rwy], 
												'etaMsgTime':etaMsgVec[ts]}, index=[0]))
											print(debug_except_notRwySw)
										print('\n')
				
	

			timeOn = ''
			timeOff = ''

			numSwitch = 0
			for k in range(1,len(meterVec)):
				if meterVec[k] != meterVec[k-1]:
					numSwitch +=1

				if meterVec[k] - meterVec[k-1] == 1:
					timeOn = timeOn + '--' + etaMsgVec[k]

				if meterVec[k] - meterVec[k-1] == -1:
					timeOff = timeOff + '--' + etaMsgVec[k]

			if numSwitch > 2:
				print(dateVec[date]  + ' ' + runwayVec[rwy] +  ' METERING FLUTTERED ON/OFF')
				totalNumberFlutter +=1

			#print(meterVec)
			
			plt.subplot(3,1,1)
			plt.plot(np.arange(len(maxActive)) , maxActive / float(60) , color = 'blue', linewidth = 5, label = 'Max Active Delay Runway ' + runwayVec[rwy])
			plt.plot(np.arange(len(maxActive)) , maxPlanned / float(60), color = 'orange', linewidth = 2, alpha = alphaVal, label = 'Max Planning Delay Runway ' + runwayVec[rwy])
			plt.plot(np.arange(len(maxActive)) , maxReady / float(60), color = 'green', linewidth = 2, alpha = alphaVal, label = 'Max Ready Delay Runway ' + runwayVec[rwy])
			plt.plot(np.arange(len(maxActive)) , meterVec , color = 'black', linewidth = 2, alpha = alphaVal, label = 'Metering ON ' + runwayVec[rwy])
			#plt.plot(np.arange(len(maxActive)) , np.cumsum(balloonMetricVec) / float(35) , color = 'red', linewidth = 2, alpha = alphaVal, label = 'Balloon Metric ' + runwayVec[rwy])
			#plt.plot(np.arange(len(maxActive)) , maxUncertain / float(60) , label = 'Max Uncertain Delay Runway ' + runwayVec[rwy])
			# plt.plot(np.arange(len(maxActive)) , meanPlannedDelay / float(60) , color = 'm',  alpha = 0.5, label = 'Mean Planned Delay Runway ' + runway)
			# plt.plot(np.arange(len(maxActive)) , minPlannedDelay / float(60) , color = 'r' , alpha = 0.5, label = 'Min Planned Delay Runway ' + runway)	
		
			plt.plot(np.arange(len(maxActive)) , upperBoundVec , '--', color = 'grey' , alpha = 0.4, linewidth = 3, label = str(upperBoundVec[-1]) + ' Minute Upper Threshold')
			plt.plot(np.arange(len(maxActive)) , targetVec , color = 'black' ,alpha = 0.4, linewidth = 3, label = str(targetVec[-1]) + ' Minute Target Queue')
			plt.plot(np.arange(len(maxActive)) , lowerBoundVec , color = 'grey' ,alpha = 0.4, linewidth = 3, label = str(lowerBoundVec[-1]) + ' Minute Lower Threshold')

			plt.ylabel('max(TTOT - UTOT) [Minutes]')
			plt.xticks(np.arange(0,len(etaMsgVec),90),displayEta,rotation=45,fontsize = 8)
			plt.title('Runway ' + runwayVec[rwy] + ' Delay ' + dateVec[date])
			plt.legend()

			plt.subplot(3,1,2)
			
			plt.plot(np.arange(len(maxActive)) , numActive , color = 'blue', linewidth = 5, label = 'Number Active Aircraft ' + runwayVec[rwy])
			plt.plot(np.arange(len(maxActive)) , numAMA , color = 'red', linewidth = 5, label = 'Number Active in AMA ' + runwayVec[rwy])
			plt.plot(np.arange(len(maxActive)) , numPlanned , color = 'orange', linewidth = 2, alpha = alphaVal, label = 'Number in Planned Group ' + runwayVec[rwy])
			plt.plot(np.arange(len(maxActive)) , numReady , color = 'green', linewidth = 2,alpha = alphaVal, label = 'Number in Ready Group ' + runwayVec[rwy])

			plt.legend()			

			plt.subplot(3,1,3)
			ax = plt.gca()
			plt.plot(np.arange(len(maxActive)) , np.zeros(len(maxActive)) , '-', color = 'black')
			plt.plot(np.arange(len(maxActive)) , np.full(len(maxActive),-2,dtype=float) , '--', color = 'black')

			idS+=1
			df_summary.loc[idS,'date'] = dateVec[date]
			df_summary.loc[idS,'runway'] = runwayVec[rwy]
			df_summary.loc[idS,'meter_switch_on_off'] = numSwitch
			df_summary.loc[idS,'meter_on'] = timeOn
			df_summary.loc[idS,'meter_off'] = timeOff
			df_summary.loc[idS,'baloon_metric'] = np.sum(balloonMetricVec)
			df_summary.loc[idS,'count_apreq'] = len(df_compliance[df_compliance['previous_state'] == 'APREQ_DEPARTURE'])
			df_summary.loc[idS,'count_edct'] = len(df_compliance[df_compliance['previous_state'] == 'EDCT_DEPARTURE'])
			df_summary.loc[idS,'count_exempt'] = len(df_compliance[df_compliance['previous_state'] == 'EXEMPT_DEPARTURE'])
			df_summary.loc[idS,'count_runway_switch'] = len(df_compliance[df_compliance['previous_state'] == 'RUNWAY_SWITCH'])
			df_summary.loc[idS,'count_planned'] = len(df_compliance[df_compliance['previous_state'] == 'PUSHBACK_PLANNED'])
			df_summary.loc[idS,'count_uncertain'] = len(df_compliance[df_compliance['previous_state'] == 'PUSHBACK_UNCERTAIN'])
			df_summary.loc[idS,'count_ready'] = len(df_compliance[df_compliance['previous_state'] == 'PUSHBACK_READY'])
			df_summary.loc[idS,'count_ga_uncertain'] = len(df_compliance[df_compliance['previous_state'] == 'GA PUSHBACK_UNCERTAIN'])
			df_summary.loc[idS,'count_ga_apreq'] = len(df_compliance[df_compliance['previous_state'] == 'GA APREQ_DEPARTURE'])
			df_summary.loc[idS,'count_ga_edct'] = len(df_compliance[df_compliance['previous_state'] == 'GA EDCT_DEPARTURE'])

			uniqueState = ['APREQ_DEPARTURE','EDCT_DEPARTURE','EXEMPT_DEPARTURE','RUNWAY_SWITCH',\
			'PUSHBACK_PLANNED','PUSHBACK_READY','PUSHBACK_UNCERTAIN','GA PUSHBACK_UNCERTAIN','GA PUSHBACK_PLANNED',\
			'GA PUSHBACK_READY', 'GA APREQ_DEPARTURE' , 'GA EDCT_DEPARTURE' , 'GA EXEMPT_DEPARTURE']

			all_compliance = []
			count_bad_compliance_5 = 0
			for state in range(len(uniqueState)):
				xPlot = []
				yPlot = []
				labelSt = uniqueState[state]
				for row in range(len(df_compliance['gufi'])):
					if df_compliance.loc[df_compliance.index[row],'previous_state'] == labelSt:
						colStr = randomColors[state]
						if str(df_compliance.loc[df_compliance.index[row],'compliance']) != 'nan':
							if df_compliance.loc[df_compliance.index[row],'compliance'] != None:
								xPlot.append(df_compliance.loc[df_compliance.index[row],'ts'])
								yPlot.append(df_compliance.loc[df_compliance.index[row],'compliance'])
								all_compliance.append(df_compliance.loc[df_compliance.index[row],'compliance'])
								if all_compliance[-1] < -5:
									count_bad_compliance_5 +=1

# TODO: set color and state, so the plot color is not randomly ordered every time
				if len(xPlot)>0:
					plt.plot(xPlot,yPlot,'*',markersize=7,color = colStr,label=labelSt)

			# print(all_compliance)
			# print(len(all_compliance))
			if len(all_compliance)>0:
				df_summary.loc[idS,'average_compliance'] = np.mean(all_compliance)
			else:
				df_summary.loc[idS,'average_compliance'] = 0
			df_summary.loc[idS,'count_non_compliant_5'] = count_bad_compliance_5

			df_compliance.to_csv('data/bank2/compliance/compliance_' +runwayVec[rwy] + '_' + dateVec[date] + '.csv')
			
			plt.legend()

			plt.tight_layout()
			if bank2:
				plt.savefig('figs/bank2/' + runwayVec[rwy] + '_' + dateVec[date] + '_bank2_delay_figV3.png')
			if bank3:
				plt.savefig('figs/bank3/' + runwayVec[rwy] + '_' + dateVec[date] + '_bank3_delay_figV3.png')

			plt.close('all')
	#plt.show()


	df_summary.to_csv('data/bank2/summary/summary_' + dateVec[date] + '.csv')
print('Total Number of Fluttering = ' + str(totalNumberFlutter))
print('writing except not-rwy-switches to file')
logbankNum = 2
if bank3:
	logbankNum = 3
debug_except_notRwySw.to_csv('debug_except_notRwySw_{}bank{}.csv'.format(daySt, logbankNum))
