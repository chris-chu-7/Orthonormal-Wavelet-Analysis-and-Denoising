% Consider the waveform extracted from the attached 'sample.wav' file. You can use `audioread` command
% in Matlab to read the signal from the file.

[data, sampleRate] = audioread('sample.wav');

% Plotting the waveform
time = linspace(0, length(data)/sampleRate, numel(data));
plot(time, data);

% a. Perform a wavelet decomposition with a Haar wavelet and then with a Daubechies-8 wavelet.
wavelet = 'haar';
level = 3;

[coefficients, levellength] = wavedec(data, level, wavelet);

wavelet2 = 'db8';

[coefficients2, levellength2] = wavedec(data, level, wavelet2);

figure;
for i = 1:level
    subplot(level + 1, 2, 2*i - 1)
    plot(wrcoef('d', coefficients, levellength, wavelet, i))
    title(['Haar wavelet Detail Coefficients Level', num2str(i)]);
    xlabel('Time(s)')
    ylabel('Detail Coefficient')

    if i == level
        subplot(level + 1, 2, 2*i)
        plot(wrcoef('a', coefficients, levellength, wavelet, i))
        title(['Haar wavelet Approximation Coefficients Level', num2str(i)]);
        xlabel('Time(s)')
        ylabel('Approximation Coefficient')
    end
end

figure;
for i = 1:level
    subplot(level + 1, 2, 2*i - 1)
    plot(wrcoef('d', coefficients2, levellength2, wavelet2, i))
    title(['Db8 wavelet Detail Coefficients Level', num2str(i)]);
    xlabel('Time(s)')
    ylabel('Detail Coefficient')

    if i == level
        subplot(level + 1, 2, 2*i)
        plot(wrcoef('a', coefficients2, levellength2, wavelet2, i))
        title(['Db8 wavelet Approximation Coefficients Level', num2str(i)]);
        xlabel('Time(s)')
        ylabel('Approximation Coefficient')
    end
end

% b. Threshold all coefficients which are about three standard deviations smaller
% than the largest coefficient (i.e. set them equal to 0). Reconstruct the signals.
% Listen to them and evaluate them both visually and acoustically (compare the two).

haarThreshold = mean(coefficients) + 3 * std(coefficients);
coefficients(abs(coefficients) < haarThreshold) = 0;

dbThreshold = mean(coefficients2) + 3 * std(coefficients2);
coefficients2(abs(coefficients2) < dbThreshold) = 0;

figure;
for i = 1:level
    subplot(level + 1, 2, 2*i - 1)
    plot(wrcoef('d', coefficients, levellength, wavelet, i))
    title(['Thresholded Haar Wavelet Detail Coefficients Level', num2str(i)]);
    xlabel('Time(s)')
    ylabel('Detail Coefficient')

    if i == level
        subplot(level + 1, 2, 2*i)
        plot(wrcoef('a', coefficients, levellength, wavelet, i))
        title(['Thresholded Haar Wavelet Approximation Coefficients Level', num2str(i)]);
        xlabel('Time(s)')
        ylabel('Approximation Coefficient')
    end
end

figure;
for i = 1:level
    subplot(level + 1, 2, 2*i - 1)
    plot(wrcoef('d', coefficients2, levellength2, wavelet2, i))
    title(['Thresholded Db8 Wavelet Detail Coefficients Level', num2str(i)]);
    xlabel('Time(s)')
    ylabel('Detail Coefficient')

    if i == level
        subplot(level + 1, 2, 2*i)
        plot(wrcoef('a', coefficients2, levellength2, wavelet2, i))
        title(['Thresholded Db8 Wavelet Approximation Coefficients Level', num2str(i)]);
        xlabel('Time(s)')
        ylabel('Approximation Coefficient')
    end
end

