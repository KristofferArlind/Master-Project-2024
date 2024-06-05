clear all
close all

all_suffix_list = {
    "_all_scale_win3std_log_2020", ...
    "_all_scale_win3std_log_2015", ...
    "_all_scale_win3std_log_2010", ...
    "_all_scale_win3std_log_2005", ...
    "_all_scale_win3std_log_2000", ...
    "_all_scale_win3std_log_1995", ...
    "_all_scale_win3std_log_1990", ...
    "_all_scale_win3std_log_1985", ...
    "_all_scale_win3std_log_1980"
    };

train_suffix_list = {
    "_train_scale_win3std_log_2020", ...
    "_train_scale_win3std_log_2015", ...
    "_train_scale_win3std_log_2010", ...
    "_train_scale_win3std_log_2005", ...
    "_train_scale_win3std_log_2000", ...
    "_train_scale_win3std_log_1995", ...
    "_train_scale_win3std_log_1990", ...
    "_train_scale_win3std_log_1985", ...
    "_train_scale_win3std_log_1980"
    };

test_suffix_list = {
    "_test_scale_win3std_log_2020", ...
    "_test_scale_win3std_log_2015", ...
    "_test_scale_win3std_log_2010", ...
    "_test_scale_win3std_log_2005", ...
    "_test_scale_win3std_log_2000", ...
    "_test_scale_win3std_log_1995", ...
    "_test_scale_win3std_log_1990", ...
    "_test_scale_win3std_log_1985", ...
    "_test_scale_win3std_log_1980"
    };

neg_features = {    
%"unemployment", "inflation" ...
     %"rate_fed_funds" "credit_spread" ...
      %  "rate_1_year", "rate_3_year", "rate_5_year", "rate_10_year"...
       %"inflation_change", "pce_change"...
       %"m1_change", "vix_change",...
       "inflation"...
    "rate_1_year_change", "rate_3_year_change", "rate_5_year_change", "rate_10_year_change"...
    "initial_claims_change", "unemployment_change" "rate_fed_funds_change" ...
    "m2_change",  "credit_spread_change"
    };

pos_features = {
 %"earnings_yield" ...
 %"ism_prod" "dvps_change" "dividend_yield_change" "eps_change",  ...
 "mc_change"...
 "real_gnp_change", "real_gdp_change"...
 "ism_prod_change", ...
 "earnings_yield_change", ...

    };

pos_features = {
     "dvps_change",
    };

%pos_features = {
%     "eps_change"
%   };

%neg_features = {
%"m1_change",
%};

neg_features = {
"vix_change",
};

AR_structure_list = {
    {NaN}, ...
    {NaN, NaN, NaN, NaN}, ...
    {NaN, NaN, NaN, NaN, NaN, NaN, NaN, NaN, NaN, NaN}
    };

AR_init_list = {
    {0.9}, ...
    {0.6, 0.2, 0.1, 0.05}, ...
    {0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.05, 0.05}
    };

AR_structure = {NaN, NaN, NaN, NaN};
AR_init = {0.6, 0.2, 0.1, 0.05};
%AR_structure = {NaN, NaN, NaN, NaN, NaN, NaN, NaN, NaN, NaN, NaN};
%AR_init = {0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.05, 0.05};

%AR_structure = NaN;
%AR_init = 0.9;



%Positively correlated with economy:
c0rec_list_pos = {-0.15, -0.1};
c0exp_list_pos = {0.15, 0.1};
%c0rec_list_pos = {-0.5};
%c0exp_list_pos = {0.5};

%Negatively correlated with economy:
c0rec_list_neg = {0.15, 0.1};
c0exp_list_neg = {-0.15, -0.1};
%c0rec_list_neg = {0.5};
%c0exp_list_neg = {-0.5};

rng(49)


use_positives = false;

for suffix_i = 1:length(all_suffix_list)
    

    test_suffix = test_suffix_list{suffix_i};
    train_suffix = train_suffix_list{suffix_i};
    all_suffix = all_suffix_list{suffix_i};

    data_files_test = {};
    data_files_train = {};
    data_files_all = {};
    
    if use_positives
        c0rec_list = c0rec_list_pos;
        c0exp_list = c0exp_list_pos;

        for i = 1:length(pos_features)
            data_files_test{end + 1} = readtable("../data/indicators/US/matlab_ready/" + pos_features{i} + test_suffix + ".csv");
            data_files_train{end + 1} = readtable("../data/indicators/US/matlab_ready/" + pos_features{i} + train_suffix + ".csv");
            data_files_all{end + 1} = readtable("../data/indicators/US/matlab_ready/" + pos_features{i} + all_suffix + ".csv");
        end
    else
        %data_files = data_files_neg;
        c0rec_list = c0rec_list_neg;
        c0exp_list = c0exp_list_neg;
    
        for i = 1:length(neg_features)
            data_files_test{end + 1} = readtable("../data/indicators/US/matlab_ready/" + neg_features{i} + test_suffix + ".csv");
            data_files_train{end + 1} = readtable("../data/indicators/US/matlab_ready/" + neg_features{i} + train_suffix + ".csv");
            data_files_all{end + 1} = readtable("../data/indicators/US/matlab_ready/" + neg_features{i} + all_suffix + ".csv");
        end
    end

    for ar_i = 1:length(AR_structure_list)
        AR_structure = AR_structure_list{ar_i};
        AR_init = AR_init_list{ar_i};

        order = length(AR_structure);
    
        for i = 1:length(data_files_train)
        
            data_train = data_files_train{i}
        
            %data = inflation;
            current_endog = data_train.Properties.VariableNames{2};
        
            if height(data_train) == 0
                disp("no train data")
                continue
            end
            
            data_train.date = datetime(data_train.date, 'InputFormat', 'yyyy-MM-dd');
            data_train.date.Format = 'yyyy-MM-dd';
        
        
            
            %Train on data before split
            %data_train = data(data.date.Year < train_test_split_year,:);
        
            
            %plot(data.date, data.unemployment_change)
            
            statenames = ["Expansion" "Recession"];
            mdl1 = arima(ARLags=1:order, Constant=NaN, Variance=NaN, AR=AR_structure, Description=statenames(1));
            mdl2 = arima(ARLags=1:order, Constant=NaN, Variance=NaN, AR=AR_structure, Description=statenames(2));
            submdl = [mdl1 mdl2];
            
            %Probability matrix
            mc = dtmc([NaN NaN; NaN NaN],StateNames=statenames);
            
            Mdl = msVAR(mc,submdl);
            
            %phi10 = 0.6;
            %phi20 = ;
            %phi30 = ;
            %phi40 = ;
            %muexp = 0.05;
            %murec = -0.05;
            %c0exp = muexp*(1 - phi10);
            %c0exp = 0.05;
            %c0rec = murec*(1 - phi10);
            %c0rec = -0.05;
        
            max_logL = -inf;
            logl = -inf;
        
            for exp_i = 1:length(c0exp_list)
                for rec_i = 1:length(c0rec_list)
        
                
                c0exp = c0exp_list{exp_i}
                c0rec = c0rec_list{rec_i}
            
                sigma_rec = 2;
                sigma_exp = 1;
                mdl10 = arima(ARLags=1:order, Constant=c0rec, AR=AR_init, Variance=sigma_rec);
                mdl20 = arima(ARLags=1:order, Constant=c0exp, AR=AR_init, Variance=sigma_exp);
                
                P0 = [0.5,0.5; 0.2, 0.8;];
                mc0 = dtmc(P0,StateNames=statenames);
                
                Mdl0 = msVAR(mc0,[mdl10 mdl20]);
                
                %figure
                %figure
                %EstMdl_train = estimate(Mdl,Mdl0,data_train.(current_endog), IterationPlot=true);
                
                [EstMdl_current, SS, logL, h] = estimate(Mdl,Mdl0,data_train.(current_endog), 'MaxIterations',1000, 'Tolerance',1e-4, IterationPlot=false);
                %title(current_endog)
                if logL > max_logL
                    max_logL = logL;
                    EstMdl = EstMdl_current;
                    best_c0rec = c0rec;
                    best_c0exp = c0exp;

                    %final_plot = iterationplot(EstMdl)
                end
                end
            end
        
            %figure
            %final_plot
            %title(current_endog)
            
            data_test = data_files_test{i};
            data_all = data_files_all{i};
        
            FS_test = filter(EstMdl, data_test.(current_endog));
            FS_all = filter(EstMdl, data_all.(current_endog));
            %FS_train = filter(EstMdl, data_train.(current_endog));
            %SS_train = smooth(EstMdl, data_train.(current_endog));
            
            %FS_all = filter(EstMdl, data.(current_endog));
            %SS_all = smooth(EstMdl, data.(current_endog));
            %FS_train = filter(EstMdl_train, data.(current_endog));
            
            %figure
            %plot(data.date(5:end), FS_train(:,1));
            %hold on
            %plot(data_all.date(order+1:end), FS_all(:,1));
            %hold on
            %figure
            %plot(data_train.date(order+1:end), FS_train(:,1));
            %hold on
            %plot(data.date(order+1:end), SS_all(:,1));
            %title(current_endog)
            %figure
            %plot(data_test.date(order+1:end), FS_test(:,1));
            %title(current_endog)
            
            test_results = table;
            test_results.date = data_test.date(order+1:end);
            test_results.p = FS_test(:,1);
            
            all_results = table;
            all_results.date = data_all.date(order+1:end);
            all_results.p = FS_all(:,1);
        
            %train_results = table;
            %train_results.date = data_train.date(order+1:end);
            %train_results.p = FS_train(:,1);
        
            %train_results_smooth = table;
            %train_results_smooth.date = data_train.date(order+1:end);
            %train_results_smooth.p = SS_train(:,1);
        
            %all_results_smooth = table;
            %all_results_smooth.date = data.date(order+1:end);
            %all_results_smooth.p = SS_all(:,1);
        
            writetable(test_results, "../results/regime/markov_matlab/" + current_endog + test_suffix + "_order" + string(order) + ".csv")
            writetable(all_results, "../results/regime/markov_matlab/" + current_endog + all_suffix + "_order" + string(order) + ".csv")
            %writetable(train_results, "../results/regime/markov_matlab/" + current_endog + train_suffix + "_order" + string(order) + ".csv")
            %writetable(train_results_smooth, "../results/regime/markov_matlab/" + current_endog + train_suffix + "_order" + string(order) + "_smooth.csv")
        
            %save("../results/regime/markov_models/" + current_endog + train_suffix + "_order" + string(order) + ".mat", "EstMdl", "max_logL", "AR_init", "best_c0exp", "best_c0rec")
            %writetable(all_results, "../results/regime/markov_matlab/" + current_endog + "_train_" + string(train_test_split_year) + "_order" + string(order) + ".csv")
            %writetable(all_results, "../results/regime/markov_matlab/" + current_endog + "_all_order" + string(order) + ".csv")
            %writetable(all_results_smooth, "../results/regime/markov_matlab/" + current_endog + "_train_" + string(train_test_split_year) + "_order" + string(order) + "_smooth" + ".csv")
        
        end
    end
end
