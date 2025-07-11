DATA_PATH           = 'data_total_extended.csv';
PREDICT_YEAR        = 2018;
WAVELET             = 'db4';
DECOMPOSITION_LEVEL = 5;

opts = detectImportOptions(DATA_PATH,'VariableNamingRule','preserve');
opts = setvartype(opts,'timestamp','datetime');
opts.SelectedVariableNames = {'timestamp','value'};
tbl = readtable(DATA_PATH,opts);

tt = table2timetable(tbl,'RowTimes','timestamp');
tt = retime(tt,'hourly','mean');
tt = rmmissing(tt);

train = tt(year(tt.Time)>=PREDICT_YEAR-3 & year(tt.Time)<PREDICT_YEAR,:);
test  = tt(year(tt.Time)==PREDICT_YEAR,:);

signal = train.value;
steps  = height(test);

[C,L] = wavedec(signal,DECOMPOSITION_LEVEL,WAVELET);

targetL = getTargetShapes(steps,DECOMPOSITION_LEVEL,WAVELET);

coeffs = cell(DECOMPOSITION_LEVEL+1,1);
coeffs{1} = appcoef(C,L,WAVELET,DECOMPOSITION_LEVEL);
for k=1:DECOMPOSITION_LEVEL
    lvl = DECOMPOSITION_LEVEL-k+1;
    coeffs{k+1} = detcoef(C,L,lvl);
end

predC = [];
for i=1:numel(coeffs)
    predC = [predC; autoForecastComponent(coeffs{i},targetL(i))];
end

forecast = waverec(predC,targetL,WAVELET);
forecast = forecast(1:steps);

actual = test.value;

mae  = mean(abs(actual-forecast));
rmse = sqrt(mean((actual-forecast).^2));
mape = mean(abs((actual-forecast)./actual))*100;

figure;
plot(test.Time,actual,'k-',test.Time,forecast,'r--','LineWidth',1.2);
title('AWT Forecast vs Actual');
xlabel('Time');
ylabel('Value');
legend('Actual','Forecast');

out = timetable(test.Time,actual,forecast,'VariableNames',{'Actual','Forecast'});
writetimetable(out,sprintf('predictions_awt_%d.csv',PREDICT_YEAR));

function L = getTargetShapes(nSteps,level,wav)
    dummy = zeros(nSteps,1);
    [~,L] = wavedec(dummy,level,wav);
end

function y = autoForecastComponent(series,nAhead)
    bestAIC = inf;
    bestMdl = [];
    for p=0:5
        for d=0:1
            for q=0:10
                try
                    Mdl = arima(p,d,q);
                    [Est,~,logL] = estimate(Mdl,series,'Display','off');
                    aic = -2*logL + 2*(p+d+q);
                    if aic<bestAIC
                        bestAIC = aic;
                        bestMdl = Est;
                    end
                catch
                end
            end
        end
    end
    if ~isempty(bestMdl)
        y = forecast(bestMdl,nAhead,'Y0',series);
    else
        y = repmat(series(end),nAhead,1);
    end
end
