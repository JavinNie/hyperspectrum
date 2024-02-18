%%
%author:jiewen nie
%date:05/02/2024
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
draw_flag=true;% enable drawing

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

%% evaluation setting
%% test1. peak_position test setting
min_PeakPosition= start_wavelength_point;% nm
max_PeakPosition= end_wavelength_point;% nm
PeakPosition_samples= 100; 
% default_setting
PeakPositionTestSetting.PeakPosition= linspace(min_PeakPosition,max_PeakPosition,PeakPosition_samples); 
PeakPositionTestSetting.Resolution=10;%nm
PeakPositionTestSetting.DR_dB=0;%dB
PeakPositionTestSetting.SNR=200;%dB

%% test2.resolution test setting
min_resolution=1;%nm
max_resolution=20;%nm
resolution_samples=10;
% default_setting
ResolutionTestSetting.PeakPosition= 1351;%0.5*(end_wavelength_point+start_wavelength_point);
ResolutionTestSetting.Resolution=linspace(min_resolution,max_resolution,resolution_samples);%nm
ResolutionTestSetting.DR_dB=0;%dB
ResolutionTestSetting.SNR=200;%dB

%% test3.dynamic range test setting
min_DR_dB=-20;%0.1
max_DR_dB=200;%10
DR_samples=10;

% default_setting
DynamicRangeTestSetting.PeakPosition=1351;%0.5*(end_wavelength_point+start_wavelength_point);
DynamicRangeTestSetting.Resolution=10;%nm
DynamicRangeTestSetting.DR_dB=linspace(min_DR_dB,max_DR_dB,DR_samples);
DynamicRangeTestSetting.SNR=200;%

%% test4.SNR test setting
min_SNR_dB=0;
max_SNR_dB=50;
SNR_samples=10;
% default_setting
SnrTestSetting.PeakPosition=1351;%0.5*(end_wavelength_point+start_wavelength_point);
SnrTestSetting.Resolution=10;%nm
SnrTestSetting.DR_dB=0;%dB
SnrTestSetting.SNR=linspace(min_SNR_dB,max_SNR_dB,SNR_samples);%dB
%% *******************************
%% *******************************
%% ***Running Section*************
%% *******************************
%% *******************************
%% Evaluation 1. Matrix numerical evaluation
%% evaluation 1.1. auto-correlation half-width-half-maximum (HWHM)
%% evaluation 1.2. mean cross-correlation coefficient
%% evaluation 1.3. Condition number of the encoding matrix
Matrix_evaluation_indices=matrix_evaluation(encoding_matrix,Wavelength_index, draw_flag);

%% Evaluation 2. Reconstruction performance evaluation
%% 1.生成波形
%% 1.1 分辨率测试，改1.测试波形的峰宽参数，然后看解码波形的保真度
%% 1.2 动态范围测试，改1.测试波形的幅度参数,输出解码信号，计算保真度。分辨率决定为最大分辨率15nm，噪声为0；
%% 1.3 信噪比测试，改1.测试波形的信噪比，（编码是一个线性过程，编码前后的信噪比不变）分辨率决定为15nm，测试波形幅度归一化为1；
%% 2.编码
%% 3.解码
%% 4.计算波形误差
%% note:后续可更改的子模块，比如解码算法模块，必须要规定好输入输出数据格式，并提供测试参数和变量设置

%% test1. peak_position test
FidelityFunctionCurve_PeakPosition=PeakPositionTest(PeakPositionTestSetting,encoding_matrix,Wavelength_index,decode_method,draw_flag);
% % % % output contain variable value and error result

%% test2. resolution test
FidelityFunctionCurve_Resolution=ResolutionTest(ResolutionTestSetting,encoding_matrix,Wavelength_index,decode_method,draw_flag);
% % % % output contain variable value and error result

%% test3. Dynamic range test
FidelityFunctionCurve_DynamicRange=DynamicRangeTest(DynamicRangeTestSetting,encoding_matrix,Wavelength_index,decode_method,draw_flag);
% output contain variable value and error result

%% test4. SNR test
FidelityFunctionCurve_SNR=SnrTest(SnrTestSetting,encoding_matrix,Wavelength_index,decode_method,draw_flag);
% output contain variable value and error result


