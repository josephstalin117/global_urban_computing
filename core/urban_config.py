csv = dict(
    raw_name='beijing_do_s',
    raw_file='/home/joseph/Projects/time_series_forecast/csv/Beijing_DO_s.csv',
    #raw_file='/Users/linjoseph/Projects/time_series_forecast/csv/Beijing_DO_s.csv',
    process_name='beijing_do_s_savgol',
    #process_file='/Users/linjoseph/Projects/time_series_forecast/csv/Beijing_DO_s_savgol.csv',
    process_file='/home/joseph/Projects/time_series_forecast/csv/Beijing_DO_s_savgol.csv',
)
water_setting = dict(
    water_column='DO',
    epochs=1000,
    n_in=30,
    n_out=2,
)
