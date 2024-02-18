

function WaveformArray=WaveformPostProcess(Waveform,DR_dB,SNR)

%% correct dimension
[RowWF,ColWF]=size(Waveform);
[RowDR,ColDR]=size(DR_dB);
[RowSNR,ColSNR]=size(SNR);
%% check
if ColWF>1
    error('Size of variable "Waveform" must be a col Vector')
end
if (RowDR*ColDR>1)&&(RowSNR*ColSNR>1)
    error(' Only one parameter can be set as a vector among "DR_dB" and "SNR" ')
end
if (RowDR~=1)||(RowSNR~=1)
    error('Row number of Both variable "DR_dB" and "SNR" must be 1')
end

%% process
if(ColDR~=1)&&(ColSNR==1)
    Waveform=awgn(Waveform,SNR,'measured'); 
    %% vary the amplitude
    Amp=10.^(DR_dB/20);
    WaveformArray=repmat(Waveform,1,length(Amp));
    WaveformArray=WaveformArray.*Amp;
    
elseif(ColDR==1)&&(ColSNR~=1)
    Amp=10.^(DR_dB/20);
    Waveform=Waveform*Amp;
    %% vary the noise level    
    WaveformArray=repmat(Waveform,1,length(SNR));
    for ii=1:length(SNR)
        WaveformArray(:,ii) = awgn(WaveformArray(:,ii),SNR(ii),'measured'); 
    end
else %all a scaler
    WaveformArray=awgn(10.^(DR_dB/20)*Waveform, SNR, 'measured'); 
end

%% return

end






