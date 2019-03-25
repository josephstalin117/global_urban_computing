csv = dict(
    raw_name='st_0',
    raw_file='/home/joseph/Projects/urban_data/st_0.csv',
)
model_setting = dict(
    model_columns=["inNums", "outNums", "week_day", "current_hour", "lag_yesterday_in", "lag_yesterday_out", "lag_last_week_in", "lag_last_week_out", "is_holiday"],
    epochs=1000,
    n_in=30,
    n_out=2,
)
