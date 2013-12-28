clear all

n_st = 10;

%Csig_sf = zeros(1, n_st);
%Covl_sf = zeros(1, n_st);

Csig_kl = zeros(1, n_st);
Covl_kl = zeros(1, n_st);

% Csig_sfnmf = zeros(1, n_st);
% Covl_sfnmf = zeros(1, n_st);

%Csig_cutoff = zeros(1, n_st);
%Covl_cutoff = zeros(1, n_st);

for i = 1:n_st
    file_org = strcat('bwe/', int2str(i), '_org.wav');
    %file_cutoff = strcat('bwe/', int2str(i), '_cutoff.wav');
    file_kl_rec = strcat('bwe/', int2str(i), '_kl_rec.wav');
    %file_sf_rec = strcat('bwe_spk_L20/', int2str(i), '_sf_rec.wav');
    %file_sfnmf_rec = strcat('bwe/', int2str(i), '_sfnmf_rec.wav');

%     [Csigk, ~, Covl] = composite(file_org, file_sfnmf_rec);
%     Csig_sfnmf(i) = Csigk;
%     Covl_sfnmf(i) = Covl;

%     [Csigk, ~, Covl] = composite(file_org, file_cutoff);
%     Csig_cutoff(i) = Csigk;
%     Covl_cutoff(i) = Covl;
     
   [Csigk, ~, Covl] = composite(file_org, file_kl_rec);
   Csig_kl(i) = Csigk;
   Covl_kl(i) = Covl;
   
   %[Csigk, ~, Covl] = composite(file_org, file_sf_rec);
   %Csig_sf(i) = Csigk;
   %Covl_sf(i) = Covl;
end
        
        