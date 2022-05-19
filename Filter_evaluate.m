%% 20220517
%% 抗噪测试
%% 加载滤波器矩阵5 --best-best
% load('D:\document\Research\Sprctrum\3D_data\Au_LC32_2circle.mat')
% A=Au_LC32_2circle;
% A=SpecFilters16;%滤波器矩阵，每一列是一个滤波器特性曲线
A=TT;

%% 滤波器矩阵维度
[m,n]=size(A);%行数 %列数
wavelength=[1200,1700];%波段范围
wl_point=linspace(wavelength(1),wavelength(2),m);

plot_flag=1;%绘图使能

if(plot_flag)
    figure %绘制滤波器曲线
    plot(wl_point,A)
end
%% 滤波器自身相关度
ave_corr=CM_Correlation(A,n);

%% 测试曲线生成
x=((1:m)-m/2)*4/m;
sig=test_sig_gen(m);
% m:每个滤波器特性曲线的采样点数

%% 重建测试,计算抗噪能力和重建效果

reconstruct_fun=@PINV_methods;
[SNR_lower,corr_coe_recons]=Noise_resist_test(sig,A,reconstruct_fun,wavelength,plot_flag);

disp(['滤波器自身相关度(0~1):' num2str(ave_corr)])
disp(['重建不失真的最低信噪比:(0~200):' num2str(SNR_lower)])
disp(['重建波形和目标波形相关度(1~0):' num2str(corr_coe_recons)])

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% 子函数1
function sig_sim=PINV_methods(A,B)
    sig_sim=pinv(A)*B;
end

%% 子函数2
function [SNR_limit,corr_coe_recons]=Noise_resist_test(SIG,A,reconstruct_fun,wavelength,plot_flag)
% INPUT
% sig:测试波形3D矩阵
% A:滤波器矩阵，每一列是一个滤波器特性曲线
% reconstruct_fun:重建函数

% OUTPUT
% SNR_limit:在噪声下重建波形无明显失真的所需的最低信噪比，反映滤波器组的抗噪能力，取值0~200，越小说明抗噪性能越好；
% corr_coe_recons:无噪声的重建波形与测试波形的相关系数，反映滤波器组的重建性能，取值范围小于等于1，越接近1代表重建效果越好
% 重建效果取决于滤波器，也取决于重建函数算法

% SNR=10*log10(Psignal/Pnoise),Psignal=Pnoise*10^(SNR/10)

[r,m,c]=size(SIG);
x=linspace(wavelength(1),wavelength(2),m);
%% without noise
sig_rc0=zeros(r*c,m);
sig_origin=zeros(r*c,m);
if(plot_flag)
    figure
end
for ii=1:r
    for jj=1:c
        sig=SIG(ii,:,jj)';
        B=A'*sig;
        %noise
%         B=B.*(1+randn(size(B))*1e-6);      
        sig_sim=reconstruct_fun(A',B);
        sig_sim=kalman_filter(sig_sim);

        sig_rc0((ii-1)*c+jj,:)=sig_sim;
        sig_origin((ii-1)*c+jj,:)=sig;

        if(plot_flag)
            subplot(3,4,4*(ii-1)+jj)
            plot(x,sig_sim,'r','LineWidth',2);
            hold on
            plot(x,sig,'b','LineWidth',1);
            hold off
            ylim([-0.2,1.2])
            title(['RMSE=',num2str(ceil(1000*sqrt(mean((sig-sig_sim).^2)))/1000)])
        end

    end
end
corr_coe_recons=corr2(sig_origin,sig_rc0);%%理想重建效果指标，越接近1越好

%% with noise
sig_rc_test=zeros(r*c,m);
if(plot_flag)
    figure
end
corr_coe=0;
SNR=0;%dB SNR=10*log10(Ps/Pnoise),Ps=Pnoise*10^(SNR/10)
while(corr_coe<0.99||SNR>200)
    for ii=1:r
        for jj=1:c
            sig=SIG(ii,:,jj)';
            B=A'*sig;
            %noise
            B = awgn(B,SNR); 

            sig_sim=reconstruct_fun(A',B);
            sig_sim=kalman_filter(sig_sim);
            sig_rc_test((ii-1)*c+jj,:)=sig_sim;
    
            if(plot_flag)
                subplot(3,4,4*(ii-1)+jj)
                plot(x,sig_sim,'r','LineWidth',2);
                hold on
                plot(x,sig,'b','LineWidth',1);
                hold off
                ylim([-0.2,1.2])
                title(['RMSE=',num2str(ceil(1000*sqrt(mean((sig-sig_sim).^2)))/1000)])
            end   
        end
    end
    corr_coe=corr2(sig_rc_test,sig_rc0);
    SNR=SNR+10;
end
SNR_limit=SNR-10;%波形抗噪不失真所需的最低信噪比
end

%% 子函数3
function ave_corr=CM_Correlation(CM,filter_num)
%CM : each Colum represents a filter
%filter_num: number of filters
[l,c]=size(CM);
if c==filter_num
%     pass
elseif l==filter_num
    CM=CM';
else
    error('[error]CM_Correlation: size of CM is wrong')
end

Correlation=corr(CM);
eye_ind=eye(filter_num);
Correlation(logical(eye_ind))=0;
max_corr=max(Correlation);
ave_corr=mean(max_corr);
% max_corr=max(max_corr)
end

%% 子函数4
function kf=kalman_filter(nsig)
    kf = zeros(size(nsig));
    
    x = 0; % state, x(k) = a * x(k-1) + b * u(k)
    a = 1; % state transfer matrix
    b = 0; % control matrix
    h = 1; % observer matrix
    p = 0; % estimate cov
    q = 0.002; % process cov0.002
    r = 0.01; % observer cov 0.05
    g = 0; % kalman gain
    
    gain =zeros(size(nsig));
    
    for i = 1 : length(nsig)
        x = a * x + b * nsig(i);
        p = a * p * a + q;
    
        g = p * h /(h * p * h + r);
        x = x + g * (nsig(i) - h * x);
        p = (1 - g * h) * p;
    
        kf(i) = x;
        gain(i) = g;
    end

end
% 
% t = 1:1000;
% nsig = 5 * sin(0.01 * t) + rand(1, length(t)) + randn(1, length(t)) + 5 * cos(0.05 * t + pi/1.5);
% 
% plot(t, nsig, t, kf, t, gain);
% legend("noise", "filtered", "kalman gain");
% grid minor;

%% 子函数５
function signal=test_sig_gen(m)
% 单峰，宽窄变化;2,20,50,
% 多峰，数目变化;3,5,8
% 多峰，间距变化;3峰，集中10%，50%，100%
% 大峰加尖锐小峰，两个峰均值差1:1, 1:5, 1:10;

%3*4个曲线，就是一个3*m*4的矩阵
% pre sig 3*6
% 3*4 sig

% close all
% m=501;
x=((1:m)-m/2)*4/m;

% para=[];
signal=zeros(3,m,4);

% figure

% 单峰
for jj=1:3
    mul=0;
    sigma=1/(3^jj);
    Amp=1;
    sig=mul_gaussian(mul,sigma,Amp,m);
    Amp=1/max(sig);
    sig=Amp*sig;
    
    signal(jj,:,1)=sig;%单峰宽窄

end

% 多峰数目
n_peak=[3,5,10];
for jj=1:3
    np=n_peak(jj);
    mul=((1:np)-np/2-0.5)*max(x)*2/np;
    sigma=1/10*ones(1,np);
    Amp=ones(1,np);
    sig=mul_gaussian(mul,sigma,Amp,m);

    Amp=1/max(sig);
    sig=Amp*sig;
    
    signal(jj,:,2)=sig;%多峰数目

end

% 多峰间距
gap_peak=[3,5,10];
for jj=1:3
    gp=gap_peak(jj);
    mul=((1:3)-3/2-0.5)*max(x)*1.5/gp;
    sigma=1/15*ones(1,3);   
    Amp=ones(1,3);
    sig=mul_gaussian(mul,sigma,Amp,m);
    Amp=1/max(sig);
    sig=Amp*sig;

    signal(jj,:,3)=sig;%多峰间距

end

% 大小尖峰
ratio_peak=[3,5,10];
for jj=1:3
    rp=ratio_peak(jj);
    mul=[0,0.4*max(x)];
    sigma=0.3*[1,1/rp/3];    
    Amp=[1,10];
    sig=mul_gaussian(mul,sigma,Amp,m);
    Amp=1/max(sig);
    sig=Amp*sig;

    signal(jj,:,4)=sig;%两个峰均值差1:1, 1:5, 1:10;

end
end

%% 子函数６
function sig=mul_gaussian(mul,sigma,amp,m)
x=((1:m)-m/2)*4/m;
if(length(mul)~=length(sigma))
    disp('size error')
end

n=length(mul);
sig=0;

for ii=1:n
    mul_1=mul(ii);
    sigma_1=sigma(ii);
    amp_1=amp(ii);
    sig=sig+amp_1*(1/sqrt(2*pi)*sigma_1)*exp(-(x-mul_1).*(x-mul_1)/(2*sigma_1^2));
end
end


