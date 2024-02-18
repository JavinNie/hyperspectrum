
function FidelityFunctionCurve=SnrTest(TestSetting,EncodingMatrix,Wavelength_index,decode_method,draw_flag)

%% get parameter
PeakPosition=TestSetting.PeakPosition;
Resolution=TestSetting.Resolution;
DR_dB=TestSetting.DR_dB;
SNR=TestSetting.SNR;

% generate waveform array—— wavelength * wave count
Waveform=SingularPeakWaveformGeneration(Wavelength_index,PeakPosition,Resolution);
% post process the signal
WaveformArray=WaveformPostProcess(Waveform,DR_dB,SNR);

%% 2.encode
EncodingMatrix=EncodingMatrix';
Encoded_sig=EncodingMatrix*WaveformArray;

%% 3.decode
Decoded_sig=decode_method(EncodingMatrix,Encoded_sig);
% [samples,counts]
%% 4.fidelity calculation
FidelityFunctionCurve= 1-(sum((WaveformArray-Decoded_sig).^2,1)./sum(WaveformArray.^2,1)).^0.5;

%% draw
if draw_flag==1
    %% fidelity curve
    figure
    plot(SNR,FidelityFunctionCurve)    
    xlabel('SNR/dB')
    ylabel('Fidelity')
    title(sprintf('Fidelity function of SNR \n *PeakPosition= %d nm,Resolution= %d nm',PeakPosition,Resolution))
    exportgraphics(gca, 'Fidelity function of SNR.tif', 'Resolution', 300);
    %% slice check 1
    figure
    index=find(FidelityFunctionCurve==max(FidelityFunctionCurve));
    plot(Wavelength_index,Waveform)
    hold on;
    plot(Wavelength_index,Decoded_sig(:,index(1)));
    hold off    
    title(sprintf('Best fidelity= %d \n *PeakPosition= %d nm,Resolution= %d nm',max(FidelityFunctionCurve),PeakPosition,Resolution))
    legend('GT','Reconstructed')
    exportgraphics(gca, 'Fidelity function of SNR BEST.tif', 'Resolution', 300);
    %% slice check 2
    figure
    index=find(FidelityFunctionCurve==min(FidelityFunctionCurve));
    plot(Wavelength_index,Waveform)
    hold on;
    plot(Wavelength_index,Decoded_sig(:,index(1)));
    hold off
    title(sprintf('Worst fidelity= %d \n *PeakPosition= %d nm,Resolution= %d nm',min(FidelityFunctionCurve),PeakPosition,Resolution))
    legend('GT','Reconstructed')
    exportgraphics(gca, 'Fidelity function of SNR WORST.tif', 'Resolution', 300);
end

end

