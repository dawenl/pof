clear all

%rir = 'booth';
dep = '_dep';
%dep = '';

%data_dir = strcat('reverb_', rir);
data_dir = strcat('../iPhone');

n_spk = 6;
n_sent = 1;

stoi_rev = zeros(n_spk, n_sent);

stoi_dr_em_np = zeros(n_spk, n_sent);

stoi_dr_me_np = zeros(n_spk, n_sent);

stoi_dr_emp = zeros(n_spk, n_sent);

stoi_cmn = zeros(n_spk, n_sent);


fprintf('\n************STOI**********\n');
for i = 1:n_spk
    for j = 1:n_sent
        file_org = strcat(data_dir, '/spk', int2str(i), '_sent', ...
            int2str(j), '_org.wav');
        file_rev = strcat(data_dir, '/spk', int2str(i), '_sent', ...
            int2str(j), '_rev.wav');
        
        file_dr_em_np = strcat(data_dir, dep, '/spk', int2str(i), '_sent', ...
            int2str(j), '_dr_np_em.wav');
        file_dr_me_np = strcat(data_dir, dep, '/spk', int2str(i), '_sent', ...
            int2str(j), '_dr_np_me.wav');
        
        file_dr_emp = strcat(data_dir, '/spk', int2str(i), '_sent', ...
            int2str(j), '_dr_emp.wav');
        
        file_cmn = strcat(data_dir, dep, '/spk', int2str(i), '_sent', ...
            int2str(j), '_dr_cmn.wav');
        
        [wav, fs] = wavread(file_org);
        [wav_rev, ~] = wavread(file_rev);
        
        [wav_dr_em_np, ~] = wavread(file_dr_em_np);
        [wav_dr_me_np, ~] = wavread(file_dr_me_np);
        
        [wav_dr_emp, ~] = wavread(file_dr_emp);
        [wav_cmn, ~] = wavread(file_cmn);
       
        stoi_rev(i, j) = stoi(wav, wav_rev, fs);
        stoi_cmn(i, j) = stoi(wav, wav_cmn, fs);
        stoi_dr_em_np(i, j) = stoi(wav, wav_dr_em_np, fs);
        stoi_dr_me_np(i, j) = stoi(wav, wav_dr_me_np, fs);
        stoi_dr_emp(i, j) = stoi(wav, wav_dr_emp, fs);
        
        fprintf('********Speaker %d Sentence %d:*********\n', i, j);
        fprintf('Reverberant speech: %.2f\n',stoi_rev(i, j)); 
        fprintf('Dereverb with CMN: %.2f\n', stoi_cmn(i, j)); 

        fprintf('Dereverb with noisy phase (EM): %.2f\n', stoi_dr_em_np(i, j)); 
        fprintf('Dereverb with noisy phase (ME): %.2f\n', stoi_dr_me_np(i, j)); 

        fprintf('Dereverb with emp filter: %.2f\n\n', stoi_dr_emp(i, j)); 
    end
end

n = length(stoi_rev);
fprintf('*********On average********\n')
fprintf('Reverberant speech\t%.2f / %.2f / %.2f\n', mean(stoi_rev), median(stoi_rev), std(stoi_rev)/sqrt(n));
fprintf('Dereverb with CMN\t%.2f / %.2f / %.2f\n', mean(stoi_cmn), median(stoi_cmn), std(stoi_cmn)/sqrt(n));
fprintf('Dereverb with noisy phase (EM)\t%.2f / %.2f / %.2f\n', mean(stoi_dr_em_np), median(stoi_dr_em_np), std(stoi_dr_em_np)/sqrt(n));
fprintf('Dereverb with noisy phase (ME)\t%.2f / %.2f / %.2f\n', mean(stoi_dr_me_np), median(stoi_dr_me_np), std(stoi_dr_me_np)/sqrt(n));
fprintf('Dereverb with emp filter\t%.2f / %.2f / %.2f\n', mean(stoi_dr_emp), median(stoi_dr_emp), std(stoi_dr_emp)/sqrt(n));

        