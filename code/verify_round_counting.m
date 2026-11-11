% verify_round_counting.m
% Reference checks for the synchronous power-domination round-counting convention.
% Place this file in the same folder as power_domination.m, then run:
%   verify_round_counting

scriptDir = fileparts(mfilename('fullpath'));
addpath(fullfile(scriptDir, 'functions'));

fprintf('Checking synchronous round-counting reference cases...\n\n');

%% Example 1: Star K_{1,4}, center selected
G = graph([1 1 1 1],[2 3 4 5]);
S = 1;
[isPDS, ppt] = power_domination(G,S);
fprintf('Example 1: star K_{1,4}, S={1}\n');
fprintf('Expected PPT = 1, observed PPT = %g\n\n', ppt);
assert(isPDS == true && ppt == 1, 'Example 1 failed.');

%% Example 2: Path P4, S={2}
G = path_graph_local(4);
S = 2;
[isPDS, ppt] = power_domination(G,S);
fprintf('Example 2: path P4, S={2}\n');
fprintf('Expected PPT = 2, observed PPT = %g\n\n', ppt);
assert(isPDS == true && ppt == 2, 'Example 2 failed.');

%% Example 3: Path P5, S={3}
G = path_graph_local(5);
S = 3;
[isPDS, ppt] = power_domination(G,S);
fprintf('Example 3: path P5, S={3}\n');
fprintf('Expected PPT = 2, observed PPT = %g\n\n', ppt);
assert(isPDS == true && ppt == 2, 'Example 3 failed.');

%% Example 4: Path P6, endpoint selected
G = path_graph_local(6);
S = 1;
[isPDS, ppt] = power_domination(G,S);
fprintf('Example 4: path P6, S={1}\n');
fprintf('Expected PPT = 5, observed PPT = %g\n\n', ppt);
assert(isPDS == true && ppt == 5, 'Example 4 failed.');

fprintf('All synchronous round-counting reference checks PASSED.\n');

function G = path_graph_local(n)
    s = 1:(n-1);
    t = 2:n;
    G = graph(s,t);
end
