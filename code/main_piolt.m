clear;
clc;

%% ============================================================
%  PDS Propagation Dynamics - Main Pilot
%
%  Topologies:
%       ER, BA, WS
%
%  Network sizes:
%       N = 50, 100, 200
%
%  Graphs per (Topology, N):
%       100
%
%  Candidate-set fractions:
%       5%, 10%, 15%
%
%  Strategies:
%       Random       -> 5 configurations
%       High-degree  -> 1 configuration
%       Distributed  -> 1 configuration
%
%  Total expected rows:
%       3 topologies
%       x 3 sizes
%       x 100 graphs
%       x 3 fractions
%       x (5 + 1 + 1)
%       = 18900 rows
%% ============================================================

%% Repository paths

scriptDir = fileparts(mfilename('fullpath'));
repoRoot = fileparts(scriptDir);
addpath(fullfile(scriptDir, 'functions'));

%% Reproducibility

rng(1);

%% Experimental configuration

networkSizes = [50 100 200];

avgDegree = 8;

numGraphs = 100;

fractions = [0.05 0.10 0.15];

topologies = {'ER', 'BA', 'WS'};

strategies = {'random', 'high-degree', 'distributed'};

numRandomConfigs = 5;

%% Expected number of rows

expectedRows = ...
    length(topologies) * ...
    length(networkSizes) * ...
    numGraphs * ...
    length(fractions) * ...
    (numRandomConfigs + 2);

fprintf('PDS Propagation Dynamics - Main Pilot\n');
fprintf('=====================================\n\n');

fprintf('Expected number of rows: %d\n\n', expectedRows);

%% Results storage

results = cell(expectedRows, 22);

row = 1;

%% Start timer

tic;

%% ============================================================
%  Main experiment
%% ============================================================

for t = 1:length(topologies)

    topology = topologies{t};

    fprintf('\nTopology: %s\n', topology);
    fprintf('-----------------------------\n');

    for ni = 1:length(networkSizes)

        n = networkSizes(ni);

        fprintf('N = %d\n', n);

        for graphID = 1:numGraphs

            %% ------------------------------------------------
            % Generate graph
            %% ------------------------------------------------

            switch topology

                case 'ER'

                    G = generate_ER(n, avgDegree);

                case 'BA'

                    mAttach = avgDegree / 2;

                    G = generate_BA(n, mAttach);

                case 'WS'

                    K = avgDegree;
                    beta = 0.1;

                    G = generate_WS(n, K, beta);

            end

            %% ------------------------------------------------
            % Random strategy
            %
            % Five independent configurations
            %% ------------------------------------------------

            strategy = 'random';

            for f = fractions

                for configID = 1:numRandomConfigs

                    %% Generate candidate set

                    S = generate_candidate_set( ...
                        G, f, strategy);

                    %% Extract structural features

                    F = extract_features(G, S);

                    %% Power domination

                    [isPDS, ppt, initialCoverage, ...
                        monitoredHistory] = ...
                        power_domination(G, S);

                    finalCoverage = ...
                        monitoredHistory(end) / n;

                    %% Store result

                    results(row,:) = { ...
                        graphID, ...
                        topology, ...
                        n, ...
                        F.NumEdges, ...
                        F.AvgDegree, ...
                        F.Density, ...
                        F.AvgClustering, ...
                        F.Diameter, ...
                        F.AvgPathLength, ...
                        strategy, ...
                        configID, ...
                        f, ...
                        F.SetSize, ...
                        F.ActualFraction, ...
                        F.MeanDegreeS, ...
                        F.MeanPairDistanceS, ...
                        F.MinPairDistanceS, ...
                        initialCoverage, ...
                        isPDS, ...
                        ppt, ...
                        finalCoverage, ...
                        avgDegree};

                    row = row + 1;

                end

            end

            %% ------------------------------------------------
            % Deterministic strategies:
            %
            % High-degree
            % Distributed
            %% ------------------------------------------------

            deterministicStrategies = { ...
                'high-degree', ...
                'distributed'};

            for ds = 1:length(deterministicStrategies)

                strategy = ...
                    deterministicStrategies{ds};

                configID = 1;

                for f = fractions

                    %% Generate candidate set

                    S = generate_candidate_set( ...
                        G, f, strategy);

                    %% Extract structural features

                    F = extract_features(G, S);

                    %% Power domination

                    [isPDS, ppt, initialCoverage, ...
                        monitoredHistory] = ...
                        power_domination(G, S);

                    finalCoverage = ...
                        monitoredHistory(end) / n;

                    %% Store result

                    results(row,:) = { ...
                        graphID, ...
                        topology, ...
                        n, ...
                        F.NumEdges, ...
                        F.AvgDegree, ...
                        F.Density, ...
                        F.AvgClustering, ...
                        F.Diameter, ...
                        F.AvgPathLength, ...
                        strategy, ...
                        configID, ...
                        f, ...
                        F.SetSize, ...
                        F.ActualFraction, ...
                        F.MeanDegreeS, ...
                        F.MeanPairDistanceS, ...
                        F.MinPairDistanceS, ...
                        initialCoverage, ...
                        isPDS, ...
                        ppt, ...
                        finalCoverage, ...
                        avgDegree};

                    row = row + 1;

                end

            end

            %% Progress indicator

            if mod(graphID, 5) == 0

                fprintf('   Completed graph %d / %d\n', ...
                    graphID, numGraphs);

            end

        end

    end

end

%% ============================================================
% Convert results to table
%% ============================================================

T = cell2table(results, ...
    'VariableNames', { ...
    'GraphID', ...
    'Topology', ...
    'N', ...
    'NumEdges', ...
    'AvgDegree', ...
    'Density', ...
    'AvgClustering', ...
    'Diameter', ...
    'AvgPathLength', ...
    'Strategy', ...
    'ConfigID', ...
    'RequestedFraction', ...
    'SetSize', ...
    'ActualFraction', ...
    'MeanDegreeS', ...
    'MeanPairDistanceS', ...
    'MinPairDistanceS', ...
    'C0', ...
    'IsPDS', ...
    'PPT', ...
    'FinalCoverage', ...
    'TargetAvgDegree'});

%% ============================================================
% Validation
%% ============================================================

fprintf('\nValidating results...\n');

assert(height(T) == expectedRows, ...
    'ERROR: Unexpected number of result rows.');

assert(~any(isnan(T.AvgDegree)), ...
    'ERROR: NaN detected in AvgDegree.');

assert(~any(isnan(T.Density)), ...
    'ERROR: NaN detected in Density.');

assert(~any(isnan(T.AvgClustering)), ...
    'ERROR: NaN detected in AvgClustering.');

assert(~any(isnan(T.Diameter)), ...
    'ERROR: NaN detected in Diameter.');

assert(~any(isnan(T.AvgPathLength)), ...
    'ERROR: NaN detected in AvgPathLength.');

assert(~any(isnan(T.MeanDegreeS)), ...
    'ERROR: NaN detected in MeanDegreeS.');

assert(~any(isnan(T.MeanPairDistanceS)), ...
    'ERROR: NaN detected in MeanPairDistanceS.');

assert(~any(isnan(T.MinPairDistanceS)), ...
    'ERROR: NaN detected in MinPairDistanceS.');

assert(~any(isnan(T.C0)), ...
    'ERROR: NaN detected in C0.');

assert(all(T.C0 >= 0 & T.C0 <= 1), ...
    'ERROR: Invalid C0 value.');

assert(all(T.FinalCoverage >= 0 & ...
           T.FinalCoverage <= 1), ...
    'ERROR: Invalid FinalCoverage value.');

%% ============================================================
% Basic summary
%% ============================================================

successfulCases = sum(T.IsPDS);

failedCases = sum(~T.IsPDS);

successRate = ...
    100 * successfulCases / height(T);

fprintf('\nPilot completed successfully.\n');
fprintf('=============================\n');

fprintf('Total rows:            %d\n', height(T));
fprintf('Successful PDS cases:  %d\n', successfulCases);
fprintf('Failed PDS cases:      %d\n', failedCases);
fprintf('Overall success rate:  %.2f%%\n', successRate);

%% ============================================================
% Runtime
%% ============================================================

elapsedTime = toc;

fprintf('Runtime:               %.2f seconds\n', ...
    elapsedTime);

%% ============================================================
% Save results
%% ============================================================

outputFile = fullfile(repoRoot, 'data', 'pds_propagation_final_results.csv');

writetable(T, outputFile);

fprintf('\nResults saved to:\n');
fprintf('%s\n', outputFile);

fprintf('\nMain Pilot PASSED.\n');