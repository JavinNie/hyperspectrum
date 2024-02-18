
function FidelityFunctionCurve=ResolutionTest(TestSetting,EncodingMatrix,Wavelength_index,decode_method,draw_flag)
%% 1.生成波形
%% 1.1 分辨率测试，改1.测试波形的峰宽参数，然后看解码波形的保真度
%% 1.2 动态范围测试，改1.测试波形的幅度参数,输出解码信号，计算保真度。分辨率决定为最大分辨率15nm，噪声为0；
%% 1.3 信噪比测试，改1.测试波形的信噪比，（编码是一个线性过程，编码前后的信噪比不变）分辨率决定为15nm，测试波形幅度归一化为1；
% 
% PeakPositionTestSetting.min_PeakPosition= start_wavelength_point;% nm
% PeakPositionTestSetting.max_PeakPosition= end_wavelength_point;% nm
% PeakPositionTestSetting.PeakPosition_samples= 10; 
% % default_setting
% PeakPositionTestSetting.Resolution=10;%nm
% PeakPositionTestSetting.DR_dB=0;%dB
% PeakPositionTestSetting.SNR=200;%dB

%% get parameter
PeakPosition=TestSetting.PeakPosition;
Resolution=TestSetting.Resolution;
DR_dB=TestSetting.DR_dB;
SNR=TestSetting.SNR;

% generate waveform array—— wavelength * wave count
WaveformArray=SingularPeakWaveformGeneration(Wavelength_index,PeakPosition,Resolution);

%% 2.编码
EncodingMatrix=EncodingMatrix';
Encoded_sig=EncodingMatrix*WaveformArray;

%% 3.解码
Decoded_sig=decode_method(EncodingMatrix,Encoded_sig);
% [samples,counts]
%% 4.计算波形误差
FidelityFunctionCurve= 1-(sum((WaveformArray-Decoded_sig).^2,1)./sum(WaveformArray.^2,1)).^0.5;

%% draw
if draw_flag==1
    %% fidelity curve
    figure
    plot(Resolution,FidelityFunctionCurve)    
    xlabel('Resolution/nm')
    ylabel('Fidelity')
    title(['Fidelity function of Resolution (PeakPosition=',num2str(PeakPosition),'nm)'])
    exportgraphics(gca, 'Fidelity function of Resolution.tif', 'Resolution', 300);
    %% slice check 1
    figure
    index=find(FidelityFunctionCurve==max(FidelityFunctionCurve));
    plot(Wavelength_index,WaveformArray(:,index(1)))
    hold on;
    plot(Wavelength_index,Decoded_sig(:,index(1)));
    hold off
    title(['Best fidelity=',num2str(max(FidelityFunctionCurve)),'(PeakPosition=',num2str(PeakPosition),'nm)'])
    legend('GT','Reconstructed')    
    exportgraphics(gca, 'Fidelity function of Resolution BEST.tif', 'Resolution', 300);
    %% slice check 2
    figure
    index=find(FidelityFunctionCurve==min(FidelityFunctionCurve));
    plot(Wavelength_index,WaveformArray(:,index(1)))
    hold on;
    plot(Wavelength_index,Decoded_sig(:,index(1)));
    hold off
    title(['Worst fidelity=',num2str(min(FidelityFunctionCurve)),'(PeakPosition=',num2str(PeakPosition),'nm)'])
    legend('GT','Reconstructed')    
    exportgraphics(gca, 'Fidelity function of Resolution WOSRT.tif', 'Resolution', 300);
end

end

