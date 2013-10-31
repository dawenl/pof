clear all

n_speech = 60;
n_noise = 10;

SDR_sf = zeros(n_speech, n_noise);
SIR_sf = zeros(n_speech, n_noise);
SAR_sf = zeros(n_speech, n_noise);

SDR_bayes = zeros(n_speech, n_noise);
SIR_bayes = zeros(n_speech, n_noise);
SAR_bayes = zeros(n_speech, n_noise);

SDR_kl = zeros(n_speech, n_noise);
SIR_kl = zeros(n_speech, n_noise);
SAR_kl = zeros(n_speech, n_noise);

SDR_is = zeros(n_speech, n_noise);
SIR_is = zeros(n_speech, n_noise);
SAR_is = zeros(n_speech, n_noise);

for i = 1:n_speech
    for j = 1:n_noise
        load(strcat('bss/sf_s', int2str(i-1), '_n', int2str(j-1)));
        [SDR, SIR, SAR, perm] = bss_eval_sources(se, s);
        SDR_sf(i, j) = SDR(perm(1));
        SIR_sf(i, j) = SIR(perm(1));
        SAR_sf(i, j) = SAR(perm(1));
        
        load(strcat('bss/bayes_s', int2str(i-1), '_n', int2str(j-1)));
        [SDR, SIR, SAR, perm] = bss_eval_sources(se, s);
        SDR_bayes(i, j) = SDR(perm(1));
        SIR_bayes(i, j) = SIR(perm(1));
        SAR_bayes(i, j) = SAR(perm(1));
         
        load(strcat('bss/kl_s', int2str(i-1), '_n', int2str(j-1)));
        [SDR, SIR, SAR, perm] = bss_eval_sources(se, s);
        SDR_kl(i, j) = SDR(perm(1));
        SIR_kl(i, j) = SIR(perm(1));
        SAR_kl(i, j) = SAR(perm(1));
        
        load(strcat('bss/is_s', int2str(i-1), '_n', int2str(j-1)));
        [SDR, SIR, SAR, perm] = bss_eval_sources(se, s);
        SDR_is(i, j) = SDR(perm(1));  
        SIR_is(i, j) = SIR(perm(1));
        SAR_is(i, j) = SAR(perm(1));
    end
end
        
        
