% function plot_burgers_results
clc
clear
close all;



% ======================================
Data = load('../Data/eq_data_N100_r2500.mat');
time_grid = Data.Time;
input = Data.Load;
output = Data.Response;

dT = 0.02;
Fs = 1/dT; 
L = length(time_grid);
n = 2^nextpow2(L);
freq = 0:(Fs/n):(Fs/2-Fs/n);
freq_index = 0:length(freq);



% ======================================
base_net = 'FNO';
train_res = 2500;
run_index = 0;
train_resolution = ['TrainRes_', num2str(train_res)];
results_dir = ['../results/',train_resolution, '/',num2str(run_index),'/'];
results_file = ['earthquake_test.mat'];
RESULTS = load([results_dir, results_file]);
X_test = RESULTS.x_test;
Y_test = RESULTS.y_test;
Y_pred = RESULTS.y_pred;


X_test_fft = fft(X_test,n,2);
X_test_fft = abs(X_test_fft/L);
X_test_fft = X_test_fft(:,1:n/2+1);
X_test_fft(:,2:end-1) = 2*X_test_fft(:,2:end-1);
X_test_fft = X_test_fft(:,1:n/2);

Y_test_fft = fft(Y_test,n,2);
Y_test_fft = abs(Y_test_fft/L);
Y_test_fft = Y_test_fft(:,1:n/2+1);
Y_test_fft(:,2:end-1) = 2*Y_test_fft(:,2:end-1);
Y_test_fft = Y_test_fft(:,1:n/2);

Y_pred_fft = fft(Y_pred,n,2);
Y_pred_fft = abs(Y_pred_fft/L);
Y_pred_fft = Y_pred_fft(:,1:n/2+1);
Y_pred_fft(:,2:end-1) = 2*Y_pred_fft(:,2:end-1);
Y_pred_fft = Y_pred_fft(:,1:n/2);

% plot_index = 1;
% fig = figure('color',[1,1,1], 'units',...
%         'centimeters','position',[2 2 25 16]);
    
MSE = zeros(size(Y_pred,1), 1);    
L2 = zeros(size(Y_pred,1), 1);
for plot_index=1:size(Y_pred,1)
    
    y_test = squeeze(RESULTS.y_test(plot_index,:) );
    y_pred = squeeze(RESULTS.y_pred(plot_index,:) );
    L2(plot_index) = norm(  y_test(:) - y_pred(:) ) ...
                    / norm( y_test(:) ) ;
    MSE(plot_index) = mean( ( y_test - y_pred ).^2  );

%     fprintf('Index: %d, L2: %.4f\n',plot_index, L2(plot_index));
    
    if plot_index <= 5
        
        freq_index_sample = freq_index(1:1000);
        X_test_fft_sample = X_test_fft(plot_index,1:1000);
        Y_test_fft_sample = Y_test_fft(plot_index,1:1000);
        Y_pred_fft_sample = Y_pred_fft(plot_index,1:1000);
        
        figure('color',[1,1,1], 'units',...
            'centimeters','position',[2 2 28 16]);
        subplot(2,6,[1,2,3]);
        plot(time_grid, X_test(plot_index,:),'k-','LineWidth',1,'DisplayName','Input');
        ylim([-1,1]);
%         legend('FontSize',14, 'location', 'northeast', 'Interpreter','latex')
        xlabel('$t$','fontsize',14,'interpreter','latex')  
        ylabel('$u$','fontsize',14,'interpreter','latex')
        title('Time space','fontsize',14,'interpreter','latex')
        set(gca,'fontsize',14,'FontName','Times','LineWidth',1); 
        
        subplot(2,6,[5,6]);
        plot(freq_index_sample,X_test_fft_sample,'k-','LineWidth',1,'DisplayName','Input')
        xlabel('freq index','fontsize',14,'interpreter','latex')  
        ylabel('Coeff','fontsize',14,'interpreter','latex')
        title('Fourier space','fontsize',14,'interpreter','latex')
%         legend('FontSize',14, 'location', 'northeast', 'Interpreter','latex')
        set(gca,'fontsize',14,'FontName','Times','LineWidth',1); 
        
        subplot(2,6,[7,8,9]);
        plot(time_grid, Y_test(plot_index,:),'r-','LineWidth',2,'DisplayName','True output');
        hold on;
        plot(time_grid, Y_pred(plot_index,:),'b--', 'LineWidth',2,'DisplayName',[base_net]);
        ylim([-1,1]);
%         legend('FontSize',14, 'location', 'northeast', 'Interpreter','latex')
        xlabel('$t$','fontsize',14,'interpreter','latex')  
        ylabel('$S$','fontsize',14,'interpreter','latex')
        set(gca,'fontsize',14,'FontName','Times','LineWidth',1); 
        
        subplot(2,6,[11,12]);
        plot(freq_index_sample(1:400),Y_test_fft_sample(1:400),'r-','LineWidth',2,'DisplayName','True output')
        hold on;
        plot(freq_index_sample(1:400),Y_pred_fft_sample(1:400),'b--', 'LineWidth',2,'DisplayName',[base_net]);
        xlabel('freq index','fontsize',14,'interpreter','latex')  
        ylabel('Coeff','fontsize',14,'interpreter','latex')
%         legend('FontSize',14, 'location', 'northeast', 'Interpreter','latex')
        set(gca,'fontsize',14,'FontName','Times','LineWidth',1); 
        
        
%         subplot(2,1,plot_index);
%         plot(time_grid, RESULTS.x_test(plot_index,:),'k--','LineWidth',2, ...
%             'DisplayName','Input');
%         hold on;
%         plot(time_grid, RESULTS.y_test(plot_index,:),'r-','LineWidth',2, ...
%             'DisplayName','True');
%         hold on;
%         plot(time_grid, RESULTS.y_pred(plot_index,:),'b--', 'LineWidth',2, ...
%             'DisplayName',[base_net]);
%         legend('FontSize',11, 'location', 'southeast', 'Interpreter','latex')
%         set(gca,'fontsize',13,'FontName','Times','LineWidth',1); 
    end
    %clf(fig);

end

% clc
fprintf('---------------------\n mean L2 : %.4f\n', mean(L2));
fprintf(' mean MSE: %.3e\n', mean(MSE));
fprintf('---------------------\n');



