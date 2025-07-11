DATA_PATH        = 'data_total_extended.csv';
TRAIN_START      = datetime(2020,1,1,0,0,0);
TRAIN_END        = datetime(2022,12,31,23,0,0);
TEST_START       = datetime(2023,1,1,0,0,0);
TEST_END         = datetime(2023,12,31,23,0,0);
DAILY_PERIOD     = 24;
HALF_DAILY       = 12;
WEEKLY_PERIOD    = 168;
MONTHLY_PERIOD   = 720;
ANNUAL_PERIOD    = 8760;
ARLAGS           = 1:6;
MALAGS           = 1:10;
OUTPUT_FILE      = 'forecast_2023_hierarchical.csv';

tb = readtable(DATA_PATH);
tb.timestamp = datetime(tb.timestamp,'InputFormat','yyyy-MM-dd HH:mm:ss');

train_idx = tb.timestamp>=TRAIN_START & tb.timestamp<=TRAIN_END;
test_idx  = tb.timestamp>=TEST_START  & tb.timestamp<=TEST_END;

data_train       = tb.value(train_idx);
timestamps_train = tb.timestamp(train_idx);
data_test        = tb.value(test_idx);
timestamps_test  = tb.timestamp(test_idx);

mean_orig = mean(data_train);
working   = data_train;

daily_idx = mod(hour(timestamps_train),DAILY_PERIOD) + 1;
daily_avg = splitapply(@mean,working,daily_idx);
working   = working - daily_avg(daily_idx) + mean(working);

half_idx = mod(hour(timestamps_train),HALF_DAILY) + 1;
half_avg = splitapply(@mean,working,half_idx);
working  = working - half_avg(half_idx) + mean(working);

dow   = weekday(timestamps_train)-1;
how   = hour(timestamps_train);
howw  = mod(dow*24+how,WEEKLY_PERIOD) + 1;
weekly_avg = splitapply(@mean,working,howw);
working    = working - weekly_avg(howw) + mean(working);

dom   = day(timestamps_train)-1;
howm  = mod(dom*24+how,MONTHLY_PERIOD) + 1;
monthly_avg = splitapply(@mean,working,howm);
working     = working - monthly_avg(howm) + mean(working);

doy   = day(timestamps_train,'dayofyear')-1;
howa  = mod(doy*24+how,ANNUAL_PERIOD) + 1;
annual_avg = splitapply(@mean,working,howa);
working    = working - annual_avg(howa) + mean(working);

trend_win = min(ANNUAL_PERIOD,floor(numel(working)/4));
trend     = movmean(working,trend_win,'Endpoints','discard');
if numel(trend)<numel(working)
    s = floor((numel(working)-numel(trend))/2)+1;
    e = s+numel(trend)-1;
    working = working(s:e)-trend+mean(working(s:e));
    ttime   = timestamps_train(s:e);
else
    working = working-trend+mean(working);
    ttime   = timestamps_train;
end

model  = arima('ARLags',ARLAGS,'D',0,'MALags',MALAGS);
estMdl = estimate(model,working,'Display','off');
[res_f,~] = forecast(estMdl,numel(data_test),'Y0',working);

last_ts = timestamps_train(end);
ts_fc   = last_ts + hours(1:numel(data_test))';

fc = res_f;
fc = fc + (daily_avg(mod(hour(ts_fc),DAILY_PERIOD)+1)-mean_orig);
fc = fc + (half_avg(mod(hour(ts_fc),HALF_DAILY)+1)-mean(working));
dow_f = weekday(ts_fc)-1;
howf  = hour(ts_fc);
fc = fc + (weekly_avg(mod(dow_f*24+howf,WEEKLY_PERIOD)+1)-mean(working));
dom_f = day(ts_fc)-1;
fc = fc + (monthly_avg(mod(dom_f*24+howf,MONTHLY_PERIOD)+1)-mean(working));
doy_f = day(ts_fc,'dayofyear')-1;
fc = fc + (annual_avg(mod(doy_f*24+howf,ANNUAL_PERIOD)+1)-mean(working));

T = table(ts_fc,fc,'VariableNames',{'timestamp','value'});
writetable(T,OUTPUT_FILE);
