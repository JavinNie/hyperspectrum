% %%
% Wavelength_index=linspace(1200,1700,250);
% %% choice1
% PeakPosition=linspace(1200,1700,250);
% Bandwidth=5;%nm
% %% choice2
% PeakPosition=0.5*(1200+1700);
% Bandwidth=linspace(0.5,30,30);%nm


function WaveformArray=SingularPeakWaveformGeneration(Wavelength_index,PeakPosition,Bandwidth)
%% output a waveform array (wavelength,variable_count)
%% dimension1 is wavelength
%% dimension2 is related to variable parameter
%% note: Only one parameter can be set a one dimension vector among "peakpositon" and "bandwidth"
%% correct dimension
[RowWL,ColWL]=size(Wavelength_index);
[RowPP,ColPP]=size(PeakPosition);
[RowBW,ColBW]=size(Bandwidth);
%% check
if (RowPP*ColPP>1)&&(RowBW*ColBW>1)
    error(' Only one parameter can be set a one dimension vector among "peakpositon" and "bandwidth" ')
end
if (RowPP~=1)||(RowBW~=1)
    error('Row number of Both variable "peakpositon" and "bandwidth" must be 1')
end
if (RowWL>1)&&(ColWL>1)
    error('Size of variable "Wavelength_index" must be a Vector')
elseif (ColWL>1)
    Wavelength_index=Wavelength_index';
end

%% genneration based on gaussian
x=Wavelength_index;
mul=PeakPosition;
sigma=Bandwidth/2/(2*log(2))^0.5;
WaveformArray=exp(-(x-mul).^2./(2*sigma.^2));%.*(1/sqrt(2*pi)*sigma) keep Peakvalue=1 instead of Integral value=1

%% 
end