Requires python3.6 or higher

Provide start date of analysis, number of days to analyze, and bank number at the command line:
python get_scheduler_analysis_data.py yyyymmdd <opt: number of days to analyze, default 1> <opt: bank number, default 2>

Output is written to data/yyyy/mm/dd/bankN


Pre-3.1.1:
Run the following scripts for dates from 20171203 to 20180312:
getData.py
plotDelayV2.py
plot_realized_queue.py

Post-3.1.1:
Run the following scripts for dates from 20180301 to present:
get_scheduler_analysis_data.py
plot_queue_analysis.py
plot_realized_queue.py

* plot_realized_queue.py can run with either version

getDataMATM.py
Gets metered flights data from matm_flight* as backup if metered_flight* output files do not exist

createColors.py
Generate colors to use for consistent plotting. Save as randomColors.npy
