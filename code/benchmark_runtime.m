clear; clc;

%% Runtime benchmark
% Compares:
%   (1) direct power-domination simulation
%   (2) structural-feature extraction + initial-coverage computation
%
% Candidate-set construction and graph generation are NOT included in the
% timed region because the reviewer specifically asked for feature extraction
% and prediction versus direct propagation simulation.

scriptDir = fileparts(mfilename('fullpath'));
addpath(fullfile(scriptDir, 'functions'));
rng(2026);

networkSizes = [50 100 200 500];
avgDegree = 8;
topologies = {'ER','BA','WS'};
fractions = [0.05 0.10 0.15];
strategies = {'random','high-degree','distributed'};

numGraphsPerCondition = 20;
numTimingRepeats = 7;

maxRows = numel(networkSizes) * numel(topologies) * ...
    numGraphsPerCondition * numel(fractions) * numel(strategies);

rows = cell(maxRows, 12);
r = 1;

fprintf('Runtime benchmark\n');
fprintf('=================\n');

for ni = 1:numel(networkSizes)
    n = networkSizes(ni);

    for ti = 1:numel(topologies)
        topology = topologies{ti};
        fprintf('N=%d, topology=%s\n', n, topology);

        for graphID = 1:numGraphsPerCondition

            switch topology
                case 'ER'
                    G = generate_ER(n, avgDegree);
                case 'BA'
                    G = generate_BA(n, avgDegree/2);
                case 'WS'
                    G = generate_WS(n, avgDegree, 0.1);
            end

            for fi = 1:numel(fractions)
                fraction = fractions(fi);

                for si = 1:numel(strategies)
                    strategy = strategies{si};

                    % Candidate construction is intentionally outside timing.
                    S = generate_candidate_set(G, fraction, strategy);

                    % Warm-up: feature extraction.
                    extract_features(G,S);

                    % Warm-up: initial coverage C0 only.
                    monitored = false(1,n);
                    monitored(S) = true;
                    for v = S
                        Nv = neighbors(G,v);
                        monitored(Nv) = true;
                    end
                    c0 = sum(monitored) / n; %#ok<NASGU>

                    % Warm-up: full direct propagation.
                    power_domination(G,S);

                    featureTimes = zeros(numTimingRepeats,1);
                    simTimes = zeros(numTimingRepeats,1);
                    c0Times = zeros(numTimingRepeats,1);

                    for rep = 1:numTimingRepeats

                        % Structural-feature extraction.
                        t = tic;
                        extract_features(G,S);
                        featureTimes(rep) = toc(t);

                        % Initial coverage C0 only.
                        t = tic;
                        monitored = false(1,n);
                        monitored(S) = true;
                        for v = S
                            Nv = neighbors(G,v);
                            monitored(Nv) = true;
                        end
                        c0 = sum(monitored) / n; %#ok<NASGU>
                        c0Times(rep) = toc(t);

                        % Direct power-domination simulation.
                        t = tic;
                        [isPDS,ppt,~,~] = power_domination(G,S);
                        simTimes(rep) = toc(t);
                    end

                    featureMs = 1000 * median(featureTimes);
                    c0Ms = 1000 * median(c0Times);
                    featurePlusC0Ms = featureMs + c0Ms;
                    simulationMs = 1000 * median(simTimes);

                    rows(r,:) = { ...
                        n, topology, graphID, fraction, strategy, ...
                        numel(S), isPDS, ppt, ...
                        featureMs, c0Ms, featurePlusC0Ms, simulationMs};

                    r = r + 1;
                end
            end
        end
    end
end

rows = rows(1:r-1,:);

T = cell2table(rows, 'VariableNames', { ...
    'N','Topology','GraphID','Fraction','Strategy','SetSize','IsPDS','PPT', ...
    'FeatureExtraction_ms','InitialCoverage_ms','FeaturePlusC0_ms', ...
    'DirectSimulation_ms'});

writetable(T, fullfile(scriptDir, 'runtime_benchmark_raw.csv'));

% Summarize by N and topology using medians.
Gsummary = groupsummary(T, {'N','Topology'}, 'median', ...
    {'FeatureExtraction_ms','InitialCoverage_ms', ...
     'FeaturePlusC0_ms','DirectSimulation_ms'});

Gsummary.FeatureToSimulationRatio = ...
    Gsummary.median_FeaturePlusC0_ms ./ ...
    Gsummary.median_DirectSimulation_ms;

writetable(Gsummary, fullfile(scriptDir, 'runtime_benchmark_summary.csv'));

disp(Gsummary);

fprintf('\nSaved runtime_benchmark_raw.csv\n');
fprintf('Saved runtime_benchmark_summary.csv\n');
