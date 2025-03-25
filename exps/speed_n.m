% Script to test chop with half precision, multiple runs, and averaging
clear all;
clc;
pc = py.importlib.import_module('pychop');
np = py.importlib.import_module('numpy');
th = py.importlib.import_module('torch');
%torch.cuda.is_available()

addpath("../tests/chop")
pe = pyenv()
% Define matrix sizes
sizes = 2.^[7, 9, 11]; % [128, 512, 2048, 8192, 32768] % , 13, 15

% Define rounding modes supported by chop
rounding_modes = [1, 2, 3, 4, 5, 6]; % 1: nearest (even), 2: up, 3: down, 4: zero, 5: stochastic (prop) 6. stochastic (uniform)
mode_names = {'Nearest (even)', 'Up', 'Down', 'Zero', 'Stochastic (prop)', 'Stochastic (uniform)'};

% Number of runs (11 total, discard first, average last 10)
num_runs = 11;

% Initialize runtime storage (for all runs)
runtimes_all = zeros(length(sizes), length(rounding_modes), num_runs);
% Initialize average runtime storage (after discarding first run)
runtimes_avg = zeros(length(sizes), length(rounding_modes));


% Initialize runtime storage (for all runs) for pychop
runtimes_all_py = zeros(length(sizes), length(rounding_modes), num_runs);
% Initialize average runtime storage (after discarding first run) for pychop
runtimes_avg_py = zeros(length(sizes), length(rounding_modes));


% Set chop options for half precision
options.format = 'h'; % Half precision (fp16)
options.subnormal = 1; % Support subnormal numbers (optional, set to 0 to flush to zero)


pc.backend('torch');

% Loop over matrix sizes
for i = 1:length(sizes)
    n = sizes(i);
    fprintf('Testing matrix size: %d x %d\n', n, n);
    
    % Generate random matrix (single precision input as required by chop)
    A = single(rand(n, n));
    
    % Loop over rounding modes
    for j = 1:length(rounding_modes)
        options.round = rounding_modes(j);
        % ch = pc.Chop('h', rmode=j); % Same setting with chop
        ch = pc.LightChop(exp_bits=5, sig_bits=10, rmode=j);

        fprintf('  Rounding mode: %s\n', mode_names{j});
        
        % Run 11 times
        for k = 1:num_runs
            A_np = np.array(A);
            A_th = th.from_numpy(A_np); % torch array
            tic;
            A_chopped_py = ch(A_th);
            runtimes_all_py(i, j, k) = toc;
            
            tic;
            A_chopped = chop(A, options);
            runtimes_all(i, j, k) = toc;

        end
        % Compute average of runs 2 through 11 (discard first run)
        runtimes_avg(i, j) = mean(runtimes_all(i, j, 2:end));
        runtimes_avg_py(i, j) = mean(runtimes_all_py(i, j, 2:end));

    end
end

% Display results
disp('Average Runtimes (seconds, discarding first run):');
fprintf('Size\t%s\t%s\t%s\t%s\t%s\n', mode_names{:});
for i = 1:length(sizes)
    fprintf('%d\t', sizes(i));
    fprintf('%.6f\t', runtimes_avg(i, :));
    fprintf('\n');
end

% Optionally save results
save('chop_runtimes_avg.mat', 'sizes', 'rounding_modes', 'mode_names', 'runtimes_all', 'runtimes_avg');

% Create a table with sizes as the first column and runtimes for each mode as subsequent columns
csv_data = [sizes', runtimes_avg];
header = ['Size', mode_names]; % Cell array for column names

% Write to CSV file
T = array2table(csv_data, 'VariableNames', header);
writetable(T, 'chop_runtimes_avg.csv');

disp('Results saved to chop_runtimes_avg.csv');


save('chop_runtimes_avg_py.mat', 'sizes', 'rounding_modes', 'mode_names', 'runtimes_all', 'runtimes_avg_py');

% Create a table with sizes as the first column and runtimes for each mode as subsequent columns
csv_data = [sizes', runtimes_avg_py];
header = ['Size', mode_names]; % Cell array for column names

% Write to CSV file
T = array2table(csv_data, 'VariableNames', header);
writetable(T, 'chop_runtimes_avg_py.csv');

disp('Results saved to chop_runtimes_avg_py.csv');