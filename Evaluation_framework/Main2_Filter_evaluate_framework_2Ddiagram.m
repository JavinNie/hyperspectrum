%%
%author:jiewen nie
%date:16/02/2024
%function:Encoding matrix evaluation for Spectrum reconstruction
%1. achieve a decoding spectrum reconstruction framework
%1.1 reconstruction code is easily plug-in
%1.2 numbers and coefficients of normalization terms are adjustable
%2. evaluation the encoding matrix using the decoding framework
%2.1 vary the SNR of detected signals
%2.2 vary the amptitude of original sinals
%2.3 vary the minnimun number of width of spectrum peak
%3. some necessary status and coeffients of the code execution in the
%command windows.
%4. all the format of input and display of spectrum signal are represented in standard units. 
%% note: experimental writting voltage and wavelength samples,and the script will get these fundamental info from file reading.
%%
clear all
close all
% clc
%% *******************************
%% *******************************
%% Setting Section
%% *******************************
%% *******************************
%% 
%% Load encoding matrix
%% 
file_adress='filter_measured_250_71.txt';
start_wavelength_point=1200;%nm
end_wavelength_point=1700;%nm
draw_flag=false;% enable drawing

%% Load encoding matrix
encoding_matrix=LoadMatrix(file_adress);%%report the read sattus and disp size of matrix
[wavelength_samples,channels]=size(encoding_matrix);
Wavelength_index=linspace(start_wavelength_point,end_wavelength_point,wavelength_samples); %%wavelength index vector %%reading from file in next version

%%
%% Parameter setting
%%
% decode_method：decode_method=1;% 1 pinv;% 2 cvx;% 3 omp;% 4 wavelet；% 5 Tval3
% decode_method=@PINV_methods;
decode_method=@LSQ_methods;

%% *******************************
%% ***Running Section*************
%% *******************************
%% Evaluation 1. Matrix numerical evaluation
%% evaluation 1.1. auto-correlation half-width-half-maximum (HWHM)
%% evaluation 1.2. mean cross-correlation coefficient
%% evaluation 1.3. Condition number of the encoding matrix
Matrix_evaluation_indices=matrix_evaluation(encoding_matrix,Wavelength_index, draw_flag);

%% draw
figure
imagesc(Wavelength_index,[1:channels],encoding_matrix');
xlabel('Wavelength/nm')%第一坐标参数
ylabel('Channels')%第二坐标参数
title(sprintf('Encoding Filter matrixs \n ACF-HWHM=%.1e;CrossCorr=%.1e;Cond=%.1e',Matrix_evaluation_indices.acf_HWHM,Matrix_evaluation_indices.cross_correlation,Matrix_evaluation_indices.cond_number));
colorbar
exportgraphics(gca, 'EncodingFilterMatrixs.tif', 'Resolution', 300);
%% Evaluation 2. Reconstruction performance evaluation
%% note:后续可更改的子模块，比如解码算法模块，必须要规定好输入输出数据格式，并提供测试参数和变量设置

%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% step1. evaluation about the PeakPostion*ResolutionBandwidth
%% peak_position test setting
min_PeakPosition= start_wavelength_point;% nm
max_PeakPosition= end_wavelength_point;% nm
PeakPosition_samples= 100; 
%% resolution test setting
min_resolution=1;%nm
max_resolution=60;%nm
Resolution_samples=50;
%% execute 
PeakPosition=linspace(min_PeakPosition,max_PeakPosition,PeakPosition_samples); 
Resolution=linspace(min_resolution,max_resolution,Resolution_samples)';
%
FidelityFunctionCurve_Peak_Res=zeros(Resolution_samples,PeakPosition_samples);
for ii=1:Resolution_samples
    PeakPositionTestSetting.PeakPosition= PeakPosition; 
    PeakPositionTestSetting.Resolution=Resolution(ii);%nm
    PeakPositionTestSetting.DR_dB=0;%dB
    PeakPositionTestSetting.SNR=200;%dB

    %% run
    FidelityFunctionCurve_Peak_Res(ii,:)=PeakPositionTest(PeakPositionTestSetting,encoding_matrix,Wavelength_index,decode_method,draw_flag);
end
%%
formatspec='Fidelity value (PeakPosition and PeakBandwidth) average=%d, max=%d, min=%d \r\n';
fprintf(formatspec,mean(FidelityFunctionCurve_Peak_Res,'all'),max(FidelityFunctionCurve_Peak_Res,[],'all'),min(FidelityFunctionCurve_Peak_Res,[],'all'));%output the size of filter matrix
[row_max,col_max]=find(FidelityFunctionCurve_Peak_Res==max(FidelityFunctionCurve_Peak_Res,[],'all'));
%% draw
figure
imagesc(PeakPosition,Resolution,FidelityFunctionCurve_Peak_Res);
xlabel('PeakPosition/nm')%第一坐标参数
ylabel('PeakBandwidth/nm')%第二坐标参数
title(sprintf('Fidelity function of PeakPosition and PeakBandwidth \n average=%.1e, max=%.1e, min=%.1e',mean(FidelityFunctionCurve_Peak_Res,'all'),max(FidelityFunctionCurve_Peak_Res,[],'all'),min(FidelityFunctionCurve_Peak_Res,[],'all')))
colorbar
exportgraphics(gca, 'FidelityFunctionofPeakPositionandPeakBandwidth.tif', 'Resolution', 300);
%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%
%% step2. evaluation about the Dynamic Range*SNR
%% dynamic range test setting
min_DR_dB=-20;%0.1
max_DR_dB=20;%10
DR_samples=100;
%% SNR test setting
min_SNR_dB=0;
max_SNR_dB=30;
SNR_samples=100;
%% peak and resolution are set as the best in last step
best_PeakPosition=PeakPosition(col_max);
best_Resolution=Resolution(row_max);
DR_dB=linspace(min_DR_dB,max_DR_dB,DR_samples);
SNR=linspace(min_SNR_dB,max_SNR_dB,SNR_samples)';%dB
%%
%% execute 
FidelityFunctionCurve_DR_SNR=zeros(SNR_samples,DR_samples);
for ii=1:SNR_samples
    DynamicRangeTestSetting.PeakPosition= best_PeakPosition; 
    DynamicRangeTestSetting.Resolution=best_Resolution;%nm
    DynamicRangeTestSetting.DR_dB=DR_dB;%dB
    DynamicRangeTestSetting.SNR=SNR(ii);%dB
    %% run
    FidelityFunctionCurve_DR_SNR(ii,:)=DynamicRangeTest(DynamicRangeTestSetting,encoding_matrix,Wavelength_index,decode_method,draw_flag);
end
%%
formatspec='Fidelity value (DynamicRange and SNR) average=%d, max=%d, min=%d \r\n';
fprintf(formatspec,mean(FidelityFunctionCurve_DR_SNR,'all'),max(FidelityFunctionCurve_DR_SNR,[],'all'),min(FidelityFunctionCurve_DR_SNR,[],'all'));
%% draw
figure
imagesc(DR_dB,SNR,FidelityFunctionCurve_DR_SNR);
xlabel('DynamicRangeGain/dB')%第一坐标参数
ylabel('SNR/dB')%第二坐标参数
title(sprintf('Fidelity function of DynamicRange and SNR \n average=%.1e, max=%.1e, min=%.1e',mean(FidelityFunctionCurve_DR_SNR,'all'),max(FidelityFunctionCurve_DR_SNR,[],'all'),min(FidelityFunctionCurve_Peak_Res,[],'all')));
colorbar
exportgraphics(gca, 'FidelityFunctionofDynamicRangeandSNR.tif', 'Resolution', 300);
%% Output

