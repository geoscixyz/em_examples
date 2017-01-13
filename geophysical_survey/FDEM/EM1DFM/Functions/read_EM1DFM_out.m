% function read_EM1DFM_out(work_dir,outfile)
% Function read_EM1DFM_out
% Reads in a UBC-EM1DFM *.out file and reads in details about inversion
%
% INPUT:
% work_dir: Directory where to find the outfile
% outfile: *.out file generated by em1dfm.exe at the end of inversion
%
% OUTPUT: Data structure for each sounding 
% out{:,1}     = [X,Y]
% out{:,2}     = [phid{:},beta{:},phim{:},phi{:}]
% where:
% phid(:)   = Achieved data misfit for each iteration
% phim(:)   = Norm of model objective function for each iteration
% phi(:)    = Norm of the objective function
% beta(:)   = Trade-off parameter per iteration

%% FOR DEV ONLY
clear all
close all

work_dir = 'C:\Users\dominiquef\Dropbox\KrisDom\4Kris\EM1DFM_test';
outfile     = 'em1dfm.out';

%% CODE START HERE
iter_count = 1;

fid = fopen([work_dir '\' outfile],'r+');
line = fgets(fid);

% Iterate until end of file
while line~=-1

    % Look for flags
    nsnd = regexp(line,'Number of soundings','match');
    iter = regexp(line,'Iteration\s','match');
    snd = regexp(line,'Sounding\s','match');
    
    % If find the number of sounding
    if isempty(nsnd) == 0
        
        nsnd = str2double( regexp(line,'\d*\.?\d*','match') );
        
    end

    % If find a sounding
    if isempty(snd) == 0
        
        temp = str2double( regexp(line,'\d*\.?\d*','match') );
        snd_num = temp(1);
        
        % Save coordinate of sounding
        out{snd_num,1}(1) = temp(2); % X
        out{snd_num,1}(2) = temp(3); % Y
        
        count = 1;
    end
    
        
    % If find an iteration for sounding "snd_num"
    if isempty(iter) == 0

        invout      = regexp(line,'\d*\.?\d*','match');
        out{snd_num,2}(count,1)   = str2double(invout{1});
        out{snd_num,2}(count,2)    = str2double(invout{2});
        out{snd_num,2}(count,3)    = str2double(invout{3});
        out{snd_num,2}(count,4)    = str2double(invout{4});
        out{snd_num,2}(count,5)     = str2double(invout{5});
        
        count = count + 1;

    end
    
    line = fgets(fid);
    
end


fclose(fid);