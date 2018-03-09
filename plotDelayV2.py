import numpy as np
import pandas as pd
import psycopg2
import pandas.io.sql as psql
import matplotlib.pyplot as plt


bank2 = True
bank3 = False


#date = '2018-02-01'
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
for i in range(10,11):
	if i < 10:
		dateVec.append(daySt + '0' + str(i))
	else:
		dateVec.append(daySt + str(i))

totalNumberFlutter = 0
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
		dfMF = pd.read_csv('/home/milin/Downloads/metered_flights_' + stMF + '.csv'  , sep=',' , index_col=False)
		#dfMF = pd.read_csv('~/Documents/MeteringAnalysis/Delay/data/bank2/bank2_MATM_data_2018-02-26.csv'  , sep=',' , index_col=False)
	except:
		try:
			dfMF = pd.read_csv('~/Documents/MeteringAnalysis/Delay/data/bank2/bank2_MATM_data_'+ dateVec[date] + '.csv'  , sep=',' , index_col=False)
		except:
			metered = False

	for rwy in range(len(runwayVec)):

		dfFiltered = df[ (df['metering_mode'] == 'TIME_BASED_METERING') & (df['runway'] == runwayVec[rwy]) \
		& (df['fix'] == df['runway']) & (df['general_stream'] == 'DEPARTURE') ]
		
		dfFiltered = dfFiltered.sort_values(by=['eta_msg_time','scheduled_time'])
		dfFiltered.to_csv('data/bank2/debug/debug'+runwayVec[rwy]+dateVec[date]+'.csv')

		activeVec = []
		
		# target = dfFiltered['target_queue_buffer'].unique()
		# try:
		# 	target = target[len(target)-1] / float(60000)
		# 	upperBound =  dfFiltered['metering_display_entry_offset'].unique()
		# 	upperBound = upperBound[len(upperBound)-1] / float(60000)
		# 	lowBound = dfFiltered['metering_display_exit_offset'].unique()
		# 	lowBound = lowBound[len(lowBound)-1] / float(60000)
		# except:
		# 	pass
			
		# print(upperBound)
		# print(target)
		# print(lowBound)

		etaMsgVec0 = dfFiltered['eta_msg_time'].drop_duplicates()
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

		plt.figure(figsize = (14,10))

		for ts in range(len(etaMsgVec)):

			dfMeteredActive = dfFiltered[ (dfFiltered['eta_msg_time'] == etaMsgVec[ts] ) & (dfFiltered['schedule_priority'].isin(['NO_PRIORITY_DEPARTURE_TAXI' , 'NO_PRIORITY_DEPARTURE_TAXI_AMA' ]))]
			dfMeteredAMA = dfFiltered[ (dfFiltered['eta_msg_time'] == etaMsgVec[ts] ) & (dfFiltered['schedule_priority'] == 'NO_PRIORITY_DEPARTURE_TAXI_AMA')]
			dfMeteredGate = dfFiltered[ (dfFiltered['eta_msg_time'] == etaMsgVec[ts]) & (dfFiltered['schedule_priority'] == 'GATE_DEPARTURE_PLANNED')]
			dfMeteredUncertain = dfFiltered[ (dfFiltered['eta_msg_time'] == etaMsgVec[ts]) & (dfFiltered['schedule_priority'] == 'GATE_DEPARTURE_UNCERTAIN')]
			dfMeteredReady = dfFiltered[ (dfFiltered['eta_msg_time'] == etaMsgVec[ts]) & (dfFiltered['schedule_priority'] == 'GATE_DEPARTURE_READY')]
			#print(pd.Timedelta(dfMeteredUncertain['ttot_minus_utot'].max())/np.timedelta64(1, 's'))
			maxActive[ts] = pd.Timedelta(dfMeteredActive['ttot_minus_utot'].max()) / np.timedelta64(1, 's')
			maxPlanned[ts] = pd.Timedelta(dfMeteredGate['ttot_minus_utot'].max() ) /np.timedelta64(1, 's')
			maxUncertain[ts] = pd.Timedelta(dfMeteredUncertain['ttot_minus_utot'].max()) / np.timedelta64(1, 's')
			maxReady[ts] = pd.Timedelta(dfMeteredReady['ttot_minus_utot'].max()) / np.timedelta64(1, 's')
			meanActive[ts] = np.mean( pd.to_timedelta(dfMeteredGate['ttot_minus_utot']) / np.timedelta64(1, 's') )
			meanPlannedDelay[ts] =  np.mean( pd.to_timedelta(dfMeteredGate['ttot_minus_utot']) / np.timedelta64(1, 's') )
			numPlanned[ts] = len(np.array(pd.to_timedelta(dfMeteredGate['ttot_minus_utot'])))
			numActive[ts] = len(np.array(pd.to_timedelta(dfMeteredActive['ttot_minus_utot'])))
			numAMA[ts] = len(np.array(pd.to_timedelta(dfMeteredAMA['ttot_minus_utot'])))
			numReady[ts] = len(np.array(pd.to_timedelta(dfMeteredReady['ttot_minus_utot'])))
			dfMeter = dfFiltered[ (dfFiltered['eta_msg_time'] == etaMsgVec[ts] )]
			stRunway = str(dfMeter.loc[dfMeter.index[0],'metering_display']).split(',')
			balloonMetricVec[ts] = max([0, maxActive[ts]/float(60) - 14] )
			#print(stRunway)
			if runwayVec[rwy] in stRunway:
				meterVec[ts] = 1

			# for st in stRunway:
			# 	#print(runwayVec[rwy])
			# 	#print(st)
			# 	if str(runwayVec[rwy]) == st:
			# 		meterVec[ts] = 1

			complianceVec[ts] = 'nan'
			exemptVec[ts] = 'nan'

			if metered:
				for flight in range(len(dfMeteredActive['flight_key'])):
					if dfMeteredActive.loc[dfMeteredActive.index[flight],'flight_key'] not in activeVec:
						activeVec.append(dfMeteredActive.loc[dfMeteredActive.index[flight],'flight_key'])
						if meterVec[ts] == 1:
							dfCompliance = dfMF[ dfMF['gufi'] == dfMeteredActive.loc[dfMeteredActive.index[flight],'flight_key'] ]
							if len(dfCompliance['gufi']) > 0:
								compliance = pd.Timedelta(pd.Timestamp(dfCompliance.loc[dfCompliance.index[0],'departure_stand_actual_time']) \
								- pd.Timestamp(dfCompliance.loc[dfCompliance.index[0],'departure_stand_surface_metered_time_value_ready']) ).total_seconds() / float(60)
								print('\n')
								print(etaMsgVec[ts])
								print(dfMeteredActive.loc[dfMeteredActive.index[flight],'flight_key'])
								print(compliance)
								print('\n')
								complianceVec[ts] = compliance
								if str(compliance) == 'nan':
									exemptVec[ts] = 0


								try:
									dfLastSchedule = dfFiltered[ (dfFiltered['eta_msg_time'] == etaMsgVec[ts-1] ) \
									& (dfFiltered['flight_key'] == dfMeteredActive.loc[dfMeteredActive.index[flight],'flight_key'] )]
									lastState = dfLastSchedule.loc[dfLastSchedule.index[0],'model_schedule_state']
									lastPriority = dfLastSchedule.loc[dfLastSchedule.index[0],'schedule_priority']
									
									plt.subplot(3,1,3)
									color = 'black'
									
									if lastState == 'PUSHBACK_PLANNED':
										color = 'orange'
										markerType = '*'
									if lastState == 'PUSHBACK_READY':
										color = 'green'
										markerType = '*'
									if lastState == 'PUSHBACK_UNCERTAIN':
										color = 'grey'
										markerType = '*'

									if lastPriority == 'APREQ_DEPARTURE':
										color = 'cyan'
										markerType = 's'
									if lastPriority == 'EDCT_DEPARTURE':
										color = 'magenta'
										markerType = 's'


									if color == 'black':
										print('LOOK INTO THIS')
										print(etaMsgVec[ts-1])
										print(dfMeteredActive.loc[dfMeteredActive.index[flight]] )

									plt.plot(ts,complianceVec[ts],marker = markerType,color = color,markersize = 7)
								except:
									pass


				
			
			minPlannedDelay[ts] = pd.Timedelta(dfMeteredGate['ttot_minus_utot'].min() ) /np.timedelta64(1, 's')
			test = pd.to_timedelta(dfMeteredGate['ttot_minus_utot']) / np.timedelta64(1, 's')
			numAboveTarget[ts] = 0
			if len(test) > 0:
				for j in range(len(test)):
					if test[test.index[j]] > 720:
						numAboveTarget[ts] +=1


		numSwitch = 0
		for k in range(1,len(meterVec)):
			if meterVec[k] != meterVec[k-1]:
				numSwitch +=1

		if numSwitch > 2:
			print(dateVec[date]  + ' ' + runwayVec[rwy] +  ' METERING FLUTTERED ON/OFF')
			totalNumberFlutter +=1



		#print(meterVec)
		
		plt.subplot(3,1,1)
		plt.plot(np.arange(len(maxActive)) , maxActive / float(60) , color = 'blue', linewidth = 5, label = 'Max Active Delay Runway ' + runwayVec[rwy])
		plt.plot(np.arange(len(maxActive)) , maxPlanned / float(60), color = 'orange', linewidth = 2, alpha = alphaVal, label = 'Max Planning Delay Runway ' + runwayVec[rwy])
		plt.plot(np.arange(len(maxActive)) , maxReady / float(60), color = 'green', linewidth = 2, alpha = alphaVal, label = 'Max Ready Delay Runway ' + runwayVec[rwy])
		plt.plot(np.arange(len(maxActive)) , meterVec , color = 'black', linewidth = 2, alpha = alphaVal, label = 'Metering ON ' + runwayVec[rwy])
		plt.plot(np.arange(len(maxActive)) , np.cumsum(balloonMetricVec) / float(35) , color = 'red', linewidth = 2, alpha = alphaVal, label = 'Balloon Metric ' + runwayVec[rwy])
		#plt.plot(np.arange(len(maxActive)) , maxUncertain / float(60) , label = 'Max Uncertain Delay Runway ' + runwayVec[rwy])
		# plt.plot(np.arange(len(maxActive)) , meanPlannedDelay / float(60) , color = 'm',  alpha = 0.5, label = 'Mean Planned Delay Runway ' + runway)
		# plt.plot(np.arange(len(maxActive)) , minPlannedDelay / float(60) , color = 'r' , alpha = 0.5, label = 'Min Planned Delay Runway ' + runway)
		
		
		#plt.plot(np.arange(len(maxActive)) , numAboveTarget , label = 'Number in Planned Group Above Target')
		target = dfFiltered['target_queue_buffer'].unique()
		for j in range(len(target)):
			target = dfFiltered['target_queue_buffer'].unique()
			target = target[j] / float(60000)
			try:
				upperBound =  dfFiltered['metering_display_entry_offset'].unique()
				upperBound = upperBound[j] / float(60000)
			except:
				upperBound =  dfFiltered['metering_display_entry_offset'].unique()
				upperBound = upperBound[0] / float(60000)
			try:
				lowBound = dfFiltered['metering_display_exit_offset'].unique()
				lowBound = lowBound[j] / float(60000)
			except:
				lowBound = dfFiltered['metering_display_exit_offset'].unique()
				lowBound = lowBound[0] / float(60000)
	
			plt.plot(np.arange(len(maxActive)) , np.full(len(maxActive) , upperBound) , '--', color = 'grey' , alpha = 0.4*(j+1), linewidth = 3, label = str(upperBound) + ' Minute Upper Threshold ' + str(j+1))
			plt.plot(np.arange(len(maxActive)) , np.full(len(maxActive) , target) , color = 'black' ,alpha = 0.4*(j+1), linewidth = 3, label = str(target) + ' Minute Target Queue ' + str(j+1))
			plt.plot(np.arange(len(maxActive)) , np.full(len(maxActive) , lowBound) , color = 'grey' ,alpha = 0.4*(j+1), linewidth = 3, label = str(lowBound) + ' Minute Lower Threshold ' + str(j+1))
		


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
		plt.plot(np.arange(len(maxActive)) , np.zeros(len(maxActive)) , '-', color = 'black')
		plt.plot(np.arange(len(maxActive)) , np.full(len(maxActive),-2) , '--', color = 'black')
		#plt.plot(np.arange(len(maxActive)) , complianceVec , '*',markersize=7, color = 'blue', label = '(AOBT - TOBT) ' + runwayVec[rwy])
		plt.plot(np.arange(len(maxActive)) , exemptVec , 'o',markersize=7, color = 'black', label = 'Exempt Push Back ' + runwayVec[rwy])
		plt.legend()

		plt.tight_layout()
		if bank2:
			plt.savefig('figs/bank2/' + runwayVec[rwy] + '_' + dateVec[date] + '_bank2_delay_figV3.png')
		if bank3:
			plt.savefig('figs/bank3/' + runwayVec[rwy] + '_' + dateVec[date] + '_bank3_delay_figV3.png')

		plt.close('all')
	#plt.show()

print('Total Number of Fluttering = ' + str(totalNumberFlutter))
